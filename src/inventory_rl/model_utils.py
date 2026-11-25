"""신경망 구성, 장치 선택, 파라미터 갱신, 가중치 저장/복원 공통 함수."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


def select_device(device: str | None = None) -> torch.device:
    """명시된 장치를 우선 사용하고, 기본값은 MPS 또는 CPU이다."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)


def make_backbone(obs_dim: int, hidden: int = 64) -> nn.Sequential:
    """관측 벡터를 두 은닉층으로 변환한다: 15 → 64 → 64 (Tanh)."""
    # Linear는 Wx+b, Tanh는 비선형 활성화 함수이다.
    return nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
    )


def update_parameters(model: nn.Module, optimizer: torch.optim.Optimizer,
                      loss: torch.Tensor) -> None:
    """손실의 경사를 계산하고 신경망 파라미터를 갱신한다."""
    optimizer.zero_grad()  # 이전 경사 제거
    loss.backward()  # 손실을 파라미터에 대해 미분
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 경사 norm 제한
    optimizer.step()  # 옵티마이저로 파라미터 갱신


def save_weights(model: nn.Module, path: str | Path, key: str) -> None:
    """평가용 가중치를 저장한다. 옵티마이저 상태는 저장하지 않는다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: model.state_dict()}, path)


def load_weights(model: nn.Module, path: str | Path, key: str,
                 device: torch.device) -> None:
    """저장한 가중치를 지정된 장치에 복원한다."""
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload[key])
