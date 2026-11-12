"""REINFORCE + 가치 기준선으로 발주량 전체를 학습한다.

표 Q[상태][행동]은 쓰지 않는다. 행동이 연속 발주량 3개라서
신경망이 바로 발주 평균을 내고, 그 주변에서 가우시안으로 뽑아 쓴다.

한 줄 요약:
    1) 관측을 보고 발주량 평균 μ, 표준편차 σ, 가치 V 를 낸다.
    2) 발주 ~ Normal(μ, σ) 에서 뽑고 환경을 한 기간 진행한다.
    3) 에피소드(30기간)가 끝나면
           G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...
           A_t = G_t − V(s_t)
       로 “생각보다 잘했으면” 그 발주의 로그확률을 올린다.
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

# 학습을 시작하자마자 평균 발주가 약 20이 되게 한다.
# sigmoid(x)=20/80=0.25 를 만족하는 x 가 logit(0.25)=ln(0.25/0.75).
_INIT_ORDER_LOGIT = math.log(0.25 / 0.75)


class ActorCritic(nn.Module):
    """한 네트워크가 정책과 가치를 같이 낸다.

    공통 층(backbone)을 통과한 뒤
      mean_head  → 세 링크의 발주 평균
      value_head → “이 관측에서 앞으로 받을 보상” 예측 V(s)
    로 갈라진다. V(s) 는 정책이 아니라 기준선이다.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, max_order: float = MAX_ORDER) -> None:
        super().__init__()
        self.max_order = max_order
        # 관측 15칸 → 64 → 64. Tanh 는 -1~1 로 눌러 준다.
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)
        # 표준편차는 양수여야 해서 log_std 를 학습하고 exp 한다.
        # 시작값 exp(1)≈2.7. 발주를 평균 근처에서 조금씩 흔든다.
        self.log_std = nn.Parameter(torch.full((act_dim,), 1.0))
        # 가중치는 0, 편향만 logit. 시작 직후 평균 발주 ≈ (20, 20, 20).
        nn.init.zeros_(self.mean_head.weight)
        nn.init.constant_(self.mean_head.bias, _INIT_ORDER_LOGIT)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        # sigmoid 가 0~1, 곱하면 0~MAX_ORDER. 음수 발주가 평균으로 안 나온다.
        mean = self.max_order * torch.sigmoid(self.mean_head(hidden))
        # 너무 작으면 탐색이 없고, 너무 크면 발주가 난리가 난다.
        std = self.log_std.exp().clamp(0.05, 12.0)
        value = self.value_head(hidden).squeeze(-1)
        return mean, std, value


class ReinforceAgent:
    """한 에피소드를 통째로 모은 뒤, 그 결과로 정책을 한 번 바꾼다."""

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
        self.gamma = gamma  # 할인율. 1에 가까울수록 먼 기간 보상도 지금 발주 탓으로 본다.
        self.entropy_coef = entropy_coef  #  entrop가 클수록 발주를 더 다양하게 뽑는다.
        self.reward_scale = reward_scale  # 원본 보상 합이 수백이라, 나눠야 V 학습이 안정적이다.
        self.model = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _tensor_obs(self, obs: np.ndarray) -> torch.Tensor:
        """numpy 관측 → 스케일 → 학습 장치(GPU/MPS/CPU) 텐서."""
        return torch.as_tensor(scale_obs(obs), dtype=torch.float32, device=self.device)

    def get_action(self, obs: np.ndarray, env, deterministic: bool = True) -> np.ndarray:
        """평가용 발주.

        deterministic=True 이면 평균 μ 만 쓴다. 발표 비교 때 이렇게 한다.
        학습 중이 아니라서 기울기를 계산하지 않는다(torch.no_grad).
        """
        self.model.eval()
        with torch.no_grad():
            mean, std, _ = self.model(self._tensor_obs(obs).unsqueeze(0))
            sampled = mean[0] if deterministic else Normal(mean, std).sample()[0]
        return clip_order(sampled.cpu().numpy())

    def act_and_log(self, obs: np.ndarray) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """학습용 발주.

        평균만 쓰면 항상 같은 발주만 해서 새로운 행동을 못 찾는다.
        그래서 Normal(μ, σ) 에서 한 점을 뽑는다.

        같이 돌려주는 것:
          log_prob : 그 발주가 나올 로그확률. 나중에 “잘했으면” 이 값을 올린다.
          entropy  : 분포가 얼마나 넓은지. 너무 좁아지면 탐색이 멈춘다.
          value    : 지금 관측의 V(s). 에피소드 끝에 G_t 와 비교한다.
        """
        obs_t = self._tensor_obs(obs).unsqueeze(0)  # 배치 차원 1칸 추가
        mean, std, value = self.model(obs_t)
        dist = Normal(mean, std)
        sampled = dist.sample()
        # 링크가 3개라 로그확률도 3개. 합쳐서 “이 발주 벡터”의 로그확률로 쓴다.
        log_prob = dist.log_prob(sampled).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        action = sampled[0].detach().cpu().numpy()
        return action, log_prob, entropy, value.squeeze(0)

    def update_episode(
        self,
        log_probs: list[torch.Tensor],
        values: list[torch.Tensor],
        rewards: list[float],
        entropies: list[torch.Tensor],
    ) -> dict[str, float]:
        """에피소드 전체 보상으로 정책을 한 번 바꾼다. 여기가 REINFORCE 본체다.

        뒤에서부터 G 를 쌓는다. 마지막 기간은 G = r_T,
        그 앞은 G_t = r_t + γ G_{t+1} 이다.

        A_t = G_t − V(s_t)
          A > 0 : 생각보다 이득 → 그 발주가 더 자주 나오게
          A < 0 : 생각보다 손해 → 그 발주가 덜 나오게
        손실은 −logπ(a|s) × A 의 평균. 마이너스를 붙여 “이득인 행동의 확률을 올리는”
        방향으로 경사하강을 쓴다.
        """
        returns = []
        discounted = 0.0
        for reward in reversed(rewards):
            # 보상을 50으로 나눠 G 와 V 의 스케일을 맞춘다.
            discounted = reward / self.reward_scale + self.gamma * discounted
            returns.insert(0, discounted)  # 맨 앞에 넣어 시간 순서를 되돌린다.
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        logp = torch.cat(log_probs)
        value_t = torch.stack(values)
        entropy = torch.cat(entropies)

        # V 는 기준선일 뿐이라 정책 쪽으로 기울기가 흘러가면 안 된다. detach.
        advantages = returns_t - value_t.detach()
        policy_loss = -(logp * advantages).mean()
        # 가치 함수는 G 를 맞추도록 따로 학습한다. 맞춰야 A 가 의미 있다.
        value_loss = torch.nn.functional.mse_loss(value_t, returns_t)
        # entropy 를 크게 유지하면 발주가 한 값에 고정되는 것을 늦춘다.
        entropy_loss = -entropy.mean()
        loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "return": float(sum(rewards)),  # 로그에는 나누기 전 원래 보상 합
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(payload["model"])
