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

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from inventory_rl.env import MAX_ORDER, clip_order
from inventory_rl.obs import scale_obs

_INIT_ORDER_LOGIT = math.log(0.25 / 0.75)


class ActorCriticNet(nn.Module):
    """actor 머리(발주 평균)와 critic 머리(V)를 한 네트워크에 둔다."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, max_order: float = MAX_ORDER) -> None:
        super().__init__()
        self.max_order = max_order
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), 1.0))
        nn.init.zeros_(self.mean_head.weight)
        nn.init.constant_(self.mean_head.bias, _INIT_ORDER_LOGIT)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        mean = self.max_order * torch.sigmoid(self.mean_head(hidden))
        std = self.log_std.exp().clamp(0.05, 12.0)
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
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.reward_scale = reward_scale
        self.model = ActorCriticNet(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _tensor_obs(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(scale_obs(obs), dtype=torch.float32, device=self.device)

    def get_action(self, obs: np.ndarray, env, deterministic: bool = True) -> np.ndarray:
        """평가용. 평균 μ 만 넣는다."""
        self.model.eval()
        with torch.no_grad():
            mean, std, _ = self.model(self._tensor_obs(obs).unsqueeze(0))
            sampled = mean[0] if deterministic else Normal(mean, std).sample()[0]
        return clip_order(sampled.cpu().numpy())

    def act_and_log(self, obs: np.ndarray) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """학습용 발주. log π, entropy, V(s) 를 한 기간 업데이트에 쓴다."""
        obs_t = self._tensor_obs(obs).unsqueeze(0)
        mean, std, value = self.model(obs_t)
        dist = Normal(mean, std)
        sampled = dist.sample()
        log_prob = dist.log_prob(sampled).sum(dim=-1)
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
            _, _, next_value = self.model(self._tensor_obs(next_obs).unsqueeze(0))
            bootstrap = 0.0 if done else float(next_value.squeeze(0).item())
        td_target = reward / self.reward_scale + self.gamma * bootstrap
        td_error = td_target - value
        policy_loss = -(log_prob.squeeze() * td_error.detach())
        value_loss = td_error.pow(2)
        entropy_loss = -entropy.squeeze()
        loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "td_error": float(td_error.detach().item()),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(payload["model"])
