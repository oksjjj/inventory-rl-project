"""REINFORCE: 실제 할인 누적 보상을 이용하여 정책을 학습한다.

수업의 기본 정책 경사 식:
    θ ← θ + α G_t ∇θ log πθ(a_t|s_t)
    G_t = r_t + γr_(t+1) + γ²r_(t+2) + ...

학습에는 보상을 50으로 나누어 사용하며, 에피소드 내 정책 손실을 평균한다.
손실 = -mean(G_t × log πθ(a_t|s_t)).
환경 보상을 직접 미분하지 않고 샘플링한 행동의 로그확률밀도를 미분한다.

읽는 순서:
    ReinforceNet.__init__ → forward → act_and_log → update_episode → get_action
학습 루프는 scripts/train_reinforce.py:
    reset → [act_and_log → env.step → 로그확률밀도와 보상 저장] × 30
          → update_episode 1회
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

# 초기 평균 발주량을 20으로 설정한다.
# μ = 80 × sigmoid(z), 20/80=0.25이므로 z=logit(0.25)=log(1/3).
_INIT_ORDER_LOGIT = math.log(0.25 / 0.75)


class ReinforceNet(nn.Module):
    """가우시안 정책망: 상태별 평균 μ(s)와 링크별 표준편차 σ를 반환한다.

    πθ(a|s) = ∏_i N(a_i; μ_i(s), σ_i²)
    평균은 신경망이 관측에 따라 결정하고, 표준편차는 별도로 학습한다.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, max_order: float = MAX_ORDER) -> None:
        super().__init__()
        self.max_order = max_order

        # 관측 15개 → 은닉층 64개 → 은닉층 64개.
        # Linear는 Wx+b, Tanh는 비선형 활성화 함수다.
        # 배치 크기 B일 때 (B,15) → (B,64) → (B,64).
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # [정책의 평균 μ(s)] 세 공급 링크별 평균 발주량을 계산한다.
        # forward에서 sigmoid × MAX_ORDER를 적용하므로 평균은 0~80 범위다.
        self.mean_head = nn.Linear(hidden, act_dim)

        # [정책의 표준편차 σ] 각 링크마다 독립적인 학습 파라미터를 둔다.
        # σ는 양수여야 하므로 log_std를 학습하고 exp(log_std)로 변환한다.
        # nn.Parameter로 등록하면 Adam이 이 값도 신경망 가중치처럼 갱신한다.
        # 초기 log_std=[1,1,1] → σ≈[2.72,2.72,2.72].
        # 상태와 무관한 σ이지만 세 링크의 값은 학습 중 서로 달라질 수 있다.
        # PyTorch Normal(μ, σ)의 두 번째 인자는 분산이 아니라 표준편차다.
        self.log_std = nn.Parameter(torch.full((act_dim,), 1.0))

        # W=0, bias=logit(0.25) → 모든 상태에서 초기 평균 발주량은 20.
        # 이후 W,b가 학습되면서 상태별 발주 평균이 달라진다.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.constant_(self.mean_head.bias, _INIT_ORDER_LOGIT)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 순전파: 아직 행동을 뽑거나 가중치를 갱신하지 않는다.
        # obs=(B,15), hidden=(B,64), mean=(B,3), std=(3,).
        hidden = self.backbone(obs)
        # 예: sigmoid 출력 [0.25,0.5,0.75] → 평균 발주 [20,40,60].
        # 평균만 제한하며 샘플링한 행동 자체는 이후 clip_order로 제한한다.
        mean = self.max_order * torch.sigmoid(self.mean_head(hidden))

        # exp로 양수 표준편차를 만들고 0.05~12 범위로 제한한다.
        # 초기 평균 20, σ≈2.72이면 약 95%의 샘플은 14.7~25.3이다.
        # σ가 작으면 좁게, 크면 넓게 탐험한다. clamp 범위 밖에서는 경사가 0일 수 있다.
        # 평균과 표준편차는 정책 손실을 통해 함께 학습된다.
        std = self.log_std.exp().clamp(0.05, 12.0)
        return mean, std


class ReinforceAgent:
    """한 에피소드의 실제 보상만으로 가우시안 정책을 한 번 갱신한다."""

    def __init__(
        self,
        obs_dim: int = 15,
        act_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        reward_scale: float = 50.0,
        device: str | None = None,
    ) -> None:
        # Mac Apple Silicon은 MPS, 사용 불가 시 CPU에서 실행한다.
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.gamma = gamma
        self.reward_scale = reward_scale

        # optimizer는 backbone, mean_head, log_std를 학습한다.
        self.model = ReinforceNet(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _tensor_obs(self, obs: np.ndarray) -> torch.Tensor:
        """NumPy 관측을 정해진 상수로 나눈 후 float32 Tensor로 변환한다."""
        # obs.py의 scale_obs 사용. 예: 수요 40/80=0.5, 기간 15/30=0.5.
        # 항상 0~1에 들어가는 min-max 정규화는 아니다. 결과 shape=(15,).
        return torch.as_tensor(scale_obs(obs), dtype=torch.float32, device=self.device)

    def get_action(self, obs: np.ndarray, env, deterministic: bool = True) -> np.ndarray:
        """평가용 발주: 기본적으로 학습된 정책의 평균을 사용한다."""
        self.model.eval()
        # eval은 평가 모드, no_grad는 자동 미분 기록을 끄는 기능이다.
        with torch.no_grad():
            # (15,) → (1,15); model 호출 시 forward 실행.
            mean, std = self.model(self._tensor_obs(obs).unsqueeze(0))
            # 결정론적 평가에서는 상태별 평균, 확률적 평가에서는 샘플을 사용한다.
            sampled = mean[0] if deterministic else Normal(mean, std).sample()[0]
        # Gymnasium에 전달할 NumPy 발주 벡터 (3,), 범위 0~80.
        return clip_order(sampled.cpu().numpy())

    def act_and_log(self, obs: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        """학습용 발주량과 선택한 행동의 로그확률밀도를 반환한다.

        학습 중에는 평균만 사용하지 않고 정규분포에서 샘플링한다.
        에피소드 종료 후 학습할 수 있도록 log_prob의 미분 그래프를 유지한다.
        """
        # 1. 입력 정규화 및 배치 차원 추가: (15,) → (1,15).
        obs_t = self._tensor_obs(obs).unsqueeze(0)

        # 2. 정책 분포의 평균과 표준편차 계산.
        # mean=(1,3), std=(3,). broadcasting으로 세 분포를 구성한다.
        mean, std = self.model(obs_t)
        dist = Normal(mean, std)

        # 3. 가우시안 정책에서 행동 샘플링: (1,3).
        # 예: 평균 [20,20,20] → 샘플 [18.1,23.4,19.0].
        # sample()은 행동 자체를 미분하지 않는다.
        # REINFORCE는 고정된 샘플의 log_prob를 미분하는 score-function 방식이다.
        # 환경에 전달하기 전 train_reinforce.py에서 clip_order를 적용한다.
        sampled = dist.sample()

        # 4. 각 링크의 로그확률밀도를 합산한다.
        # 연속값 하나의 확률이 아니라 확률밀도를 사용한다.
        # π(a|s)=∏_i π_i(a_i|s) → log π(a|s)=Σ_i log π_i(a_i|s).
        # (1,3) → sum(dim=-1) → (1,).
        # 제한 전 샘플의 log_prob로 학습하며, 환경은 제한된 행동의 보상을 제공한다.
        # log_prob는 mean 및 std에 대한 경사를 유지해야 한다.
        log_prob = dist.log_prob(sampled).sum(dim=-1)

        # 5. 환경에 전달할 행동은 NumPy로 변환한다.
        # log_prob는 detach하지 않고 그대로 반환한다.
        # 이번 기간에는 학습하지 않고, 에피소드 종료까지 저장한다.
        action = sampled[0].detach().cpu().numpy()
        return action, log_prob

    def update_episode(
        self,
        log_probs: list[torch.Tensor],
        rewards: list[float],
    ) -> dict[str, float]:
        """에피소드의 Monte Carlo Return G_t로 정책을 갱신한다."""

        # 1. 실제 보상으로 시점별 할인 누적 보상 G_t 계산.
        # 30기간이라면 G_30=0에서 시작해 G_29부터 역순으로 계산한다.
        # G_t = r_t/50 + γG_(t+1). 보상 /50은 학습 신호의 크기를 조절한다.
        # 예: 보상 [10,20,30], γ=0.99이면
        # G_2=0.6, G_1=0.4+0.99×0.6=0.994,
        # G_0=0.2+0.99×0.994=1.18406.
        returns = []
        discounted = 0.0
        for reward in reversed(rewards):
            discounted = reward / self.reward_scale + self.gamma * discounted
            returns.insert(0, discounted)

        # 2. 시간 순서대로 Tensor를 구성한다. 두 Tensor 모두 (T,).
        # returns_t는 환경에서 받은 실제 보상으로 계산했으므로 상수로 취급한다.
        # log_probs는 각 시점 (1,)인 Tensor 리스트다. cat으로 (T,)를 만든다.
        # cat은 자동 미분 그래프를 유지하므로 나중에 정책의 경사를 구할 수 있다.
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        logp = torch.cat(log_probs)

        # 3. REINFORCE 정책 손실.
        # L = -mean(G_t × log πθ(a_t|s_t)).
        # G_t>0이면 해당 행동의 로그확률밀도를 높이는 방향으로,
        # G_t<0이면 낮추는 방향으로 정책 경사를 계산한다.
        # 예: G=2, logπ=-3 → L=6. logπ=-2가 되면 L=4로 감소한다.
        # - 부호는 경사상승을 경사하강으로 구현하기 위한 것이다.
        # 이 구현에서는 할인율 γ를 G_t 계산에 적용한다(별도 γ^t 가중치 없음).
        policy_loss = -(logp * returns_t).mean()

        # 4. 정책 파라미터 갱신. 에피소드당 한 번만 수행한다.
        # zero_grad: 이전 경사를 제거한다.
        self.optimizer.zero_grad()
        # backward: 각 정책 파라미터의 경사를 계산한다.
        policy_loss.backward()
        # 전체 경사 norm을 1로 제한해 큰 갱신을 완화한다.
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        # Adam이 backbone, mean_head, log_std를 실제로 갱신한다.
        self.optimizer.step()

        # 로그용 수치는 Tensor에서 꺼내며, return은 원본 보상의 할인 없는 합이다.
        return {
            "loss": float(policy_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "return": float(sum(rewards)),
        }

    def save(self, path: str | Path) -> None:
        # 학습된 정책망의 가중치와 log_std를 저장한다.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        # 저장된 정책 파라미터를 복원한다.
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(payload["model"])
