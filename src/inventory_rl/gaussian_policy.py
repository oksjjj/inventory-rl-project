"""가우시안 정책의 초기화, 분포 파라미터 계산, 샘플링.

공급 링크마다 독립 정규분포를 사용한다.
πθ(a|s) = ∏_i N(a_i; μ_i(s), σ_i²)
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from inventory_rl.env import clip_order

# 평균 20 = 80 × sigmoid(log(0.25/0.75)).
_INIT_ORDER_LOGIT = math.log(0.25 / 0.75)


def initialize_gaussian(mean_head: nn.Linear, act_dim: int) -> nn.Parameter:
    """평균 20, 표준편차 exp(1)≈2.72로 초기화한다 (발주 상한 80 기준)."""
    # 각 링크의 log_std는 상태에 무관한 학습 파라미터이다.
    log_std = nn.Parameter(torch.full((act_dim,), 1.0))
    nn.init.zeros_(mean_head.weight)
    nn.init.constant_(mean_head.bias, _INIT_ORDER_LOGIT)
    return log_std


def gaussian_parameters(hidden: torch.Tensor, mean_head: nn.Linear,
                        log_std: torch.Tensor, max_order: float
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """상태별 평균과 링크별 표준편차를 계산한다."""
    # mean=(B,3), std=(3,). 평균을 sigmoid로 0~80 범위로 제한한다.
    # 예: sigmoid 출력 [0.25,0.5,0.75] → 평균 [20,40,60].
    mean = max_order * torch.sigmoid(mean_head(hidden))
    # exp로 양수 표준편차를 얻는다. σ가 클수록 탐험 범위가 넓다.
    # 초기 평균 20에서 약 95%는 14.7~25.3. clamp 밖에서는 경사가 0일 수 있다.
    std = log_std.exp().clamp(0.05, 12.0)
    return mean, std


def sample_gaussian(mean: torch.Tensor, std: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, Normal]:
    """행동 샘플, 결합 로그확률밀도, 정규분포를 반환한다."""
    # Normal의 두 번째 인자는 분산이 아니라 표준편차이다.
    dist = Normal(mean, std)
    # sample()은 행동 자체를 미분하지 않는다. 예: [18.1,23.4,19.0].
    sampled = dist.sample()
    # 독립 분포의 결합밀도는 곱이므로 로그밀도는 합이다.
    # log π(a|s) = Σ_i log π_i(a_i|s). (B,3) → (B,).
    # 로그확률밀도의 미분 그래프를 유지해 평균과 표준편차를 학습한다.
    log_prob = dist.log_prob(sampled).sum(dim=-1)
    return sampled, log_prob, dist


def gaussian_eval_action(mean: torch.Tensor, std: torch.Tensor,
                         deterministic: bool = True) -> np.ndarray:
    """결정론적 평가에는 평균을, 확률적 평가에는 샘플을 사용한다."""
    sampled = mean[0] if deterministic else Normal(mean, std).sample()[0]
    return clip_order(sampled.cpu().numpy())
