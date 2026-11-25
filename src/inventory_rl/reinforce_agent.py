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
        self.backbone = make_backbone(obs_dim, hidden)

        # [정책의 평균 μ(s)] 세 공급 링크별 평균 발주량을 계산한다.
        # forward에서 sigmoid × MAX_ORDER를 적용하므로 평균은 0~80 범위다.
        self.mean_head = nn.Linear(hidden, act_dim)

        # 평균 출력층과 링크별 표준편차를 초기화한다.
        self.log_std = initialize_gaussian(self.mean_head, act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 순전파: 아직 행동을 뽑거나 가중치를 갱신하지 않는다.
        # obs=(B,15), hidden=(B,64), mean=(B,3), std=(3,).
        hidden = self.backbone(obs)
        mean, std = gaussian_parameters(hidden, self.mean_head, self.log_std, self.max_order)
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
        self.device = select_device(device)
        self.gamma = gamma
        self.reward_scale = reward_scale

        # optimizer는 backbone, mean_head, log_std를 학습한다.
        self.model = ReinforceNet(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, obs: np.ndarray, env, deterministic: bool = True) -> np.ndarray:
        """평가용 발주: 기본적으로 학습된 정책의 평균을 사용한다."""
        self.model.eval()
        # eval은 평가 모드, no_grad는 자동 미분 기록을 끄는 기능이다.
        with torch.no_grad():
            # (15,) → (1,15); model 호출 시 forward 실행.
            mean, std = self.model(tensor_obs(obs, self.device).unsqueeze(0))
            return gaussian_eval_action(mean, std, deterministic)

    def act_and_log(self, obs: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        """학습용 발주량과 선택한 행동의 로그확률밀도를 반환한다.

        학습 중에는 평균만 사용하지 않고 정규분포에서 샘플링한다.
        에피소드 종료 후 학습할 수 있도록 log_prob의 미분 그래프를 유지한다.
        """
        # 1. 입력 정규화 및 배치 차원 추가: (15,) → (1,15).
        obs_t = tensor_obs(obs, self.device).unsqueeze(0)

        # 2. 정책 분포의 평균과 표준편차 계산.
        # mean=(1,3), std=(3,). broadcasting으로 세 분포를 구성한다.
        mean, std = self.model(obs_t)
        # 가우시안 행동을 샘플링하고 결합 로그확률밀도를 계산한다.
        sampled, log_prob, _ = sample_gaussian(mean, std)

        # 5. 환경에 넣을 행동은 NumPy 로. log_prob 는 detach 하지 않는다.
        # 이번 기간에는 학습하지 않고, train_reinforce 가 리스트에 쌓아 둔다.
        # 에피소드가 끝나면 update_episode(log_probs, rewards) 가
        # L = −mean(G_t × log_prob_t) 에 넣어 θ 를 바꾼다. 수업 식 ∇ log π × G_t.
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

        update_parameters(self.model, self.optimizer, policy_loss)

        # 로그용 수치는 Tensor에서 꺼내며, return은 원본 보상의 할인 없는 합이다.
        return {
            "loss": float(policy_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "return": float(sum(rewards)),
        }

    def save(self, path: str | Path) -> None:
        save_weights(self.model, path, key="model")

    def load(self, path: str | Path) -> None:
        load_weights(self.model, path, key="model", device=self.device)
