"""TD actor-critic으로 발주량 전체를 학습한다. (수업 8강 슬라이드 21–22)

정책(actor)과 가치(critic)를 같이 두지만, REINFORCE와 달리
에피소드가 끝날 때까지 기다리지 않는다. 매 기간 TD 오차로 바꾼다.

8강 요약의 네 식 중 마지막, TD Actor-Critic:
    δ_t = r_t + γ V(s_{t+1}) − V(s_t)     (끝나면 γ항은 0)
    actor  : Δθ ∝ ∇ log π(a_t | s_t) × δ_t
    critic : V(s_t) 가 r_t + γ V(s_{t+1}) 을 맞추게  (7강 TD 예측과 같음)

구현하지 않은 것:
    Q Actor-Critic     → ∇ log π × Q_w(s, a)
    Advantage AC       → ∇ log π × (Q − V)

발주 분포는 REINFORCE와 같다. Normal(μ, σ), 평균은 0~80.
다른 점은 학습 신호다. REINFORCE는 에피소드 수익 G_t, 여기는 한 칸 TD.

슬라이드 18은 actor 파라미터 θ 와 critic 파라미터 w 를 따로 둔다.
여기서는 backbone 을 공유하고 머리만 나눈다. 실습용 A2C 형태다.
엔트로피 항과 보상/50 은 슬라이드에 없고 학습 안정용이다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from inventory_rl.env import MAX_ORDER
from inventory_rl.gaussian_policy import (
    initialize_gaussian, gaussian_parameters, sample_gaussian, gaussian_eval_action,
)
from inventory_rl.obs import tensor_obs
from inventory_rl.model_utils import select_device, make_backbone, update_parameters, save_weights, load_weights

class ActorCriticNet(nn.Module):
    """actor 머리(발주 평균)와 critic 머리(V)를 한 네트워크에 둔다."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, max_order: float = MAX_ORDER) -> None:
        super().__init__()
        self.max_order = max_order
        self.backbone = make_backbone(obs_dim, hidden)
        self.mean_head = nn.Linear(hidden, act_dim)  # 발주 평균 3칸
        self.value_head = nn.Linear(hidden, 1)  # V(s) 1칸
        # 평균 출력층과 링크별 표준편차를 초기화한다.
        self.log_std = initialize_gaussian(self.mean_head, act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        mean, std = gaussian_parameters(hidden, self.mean_head, self.log_std, self.max_order)
        value = self.value_head(hidden).squeeze(-1)
        return mean, std, value


class ActorCriticAgent:
    """매 기간 TD 오차 δ로 actor와 critic을 같이 바꾼다."""

    def __init__(
        self,
        obs_dim: int = 15,
        act_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        reward_scale: float = 50.0,
        device: str | None = None,
    ) -> None:
        self.device = select_device(device)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.reward_scale = reward_scale
        self.model = ActorCriticNet(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, obs: np.ndarray, env, deterministic: bool = True) -> np.ndarray:
        """평가용. 평균 μ 만 넣는다."""
        self.model.eval()
        with torch.no_grad():
            mean, std, _ = self.model(tensor_obs(obs, self.device).unsqueeze(0))
            return gaussian_eval_action(mean, std, deterministic)

    def act_and_log(self, obs: np.ndarray) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """학습용 발주. log π, entropy, V(s) 를 한 기간 업데이트에 쓴다."""
        obs_t = tensor_obs(obs, self.device).unsqueeze(0)
        mean, std, value = self.model(obs_t)
        # 가우시안 행동을 샘플링하고 결합 로그확률밀도를 계산한다.
        sampled, log_prob, dist = sample_gaussian(mean, std)
        entropy = dist.entropy().sum(dim=-1)
        action = sampled[0].detach().cpu().numpy()
        return action, log_prob, entropy, value.squeeze(0)

    def learn_step(
        self,
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> dict[str, float]:
        """한 기간 전이로 actor·critic을 한 번 바꾼다.

        목표 y = r/scale + γ V(s')   (끝나면 γ항은 0)
        δ = y − V(s)
        actor 손실 = −log π(a|s) × δ   (δ는 상수로 본다)
        critic 손실 = δ²
        """
        with torch.no_grad():
            _, _, next_value = self.model(tensor_obs(next_obs, self.device).unsqueeze(0))
            bootstrap = 0.0 if done else float(next_value.squeeze(0).item())
        # 예: r=30, V(s')=2.0, γ=0.99 → y=30/50 + 0.99*2.0 = 2.58
        td_target = reward / self.reward_scale + self.gamma * bootstrap
        td_error = td_target - value  # δ. 양수면 “생각보다 이득”
        policy_loss = -(log_prob.squeeze() * td_error.detach())  # δ는 상수로 취급
        value_loss = td_error.pow(2)
        entropy_loss = -entropy.squeeze()
        loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss

        update_parameters(self.model, self.optimizer, loss)
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "td_error": float(td_error.detach().item()),
        }

    def save(self, path: str | Path) -> None:
        save_weights(self.model, path, key="model")

    def load(self, path: str | Path) -> None:
        load_weights(self.model, path, key="model", device=self.device)
