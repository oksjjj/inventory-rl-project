"""세 강화학습 스크립트에서 사용하는 공통 실행 및 결과 저장 함수.

알고리즘마다 에피소드 내 학습 시점과 손실 계산은 다르므로
실제 학습 루프는 각 train_*.py에 그대로 둔다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from inventory_rl.env import output_dir_for_demand


def parse_train_args(description: str) -> argparse.Namespace:
    """공통 학습 실행 옵션을 정의한다."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--demand", choices=["stationary", "trend_shock"], default="trend_shock")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--log-path", type=str, default="")
    parser.add_argument("--fig-path", type=str, default="")
    return parser.parse_args()


def seed_training(seed: int) -> None:
    """모델 초기화와 행동 샘플링에 쓰는 NumPy/PyTorch 난수를 고정한다."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_paths(root: Path, args: argparse.Namespace, stem: str
                   ) -> tuple[Path, Path, Path]:
    """체크포인트, CSV 로그, 학습 곡선의 경로를 결정한다."""
    out_dir = root / (args.out_dir or output_dir_for_demand(args.demand))
    save_path = root / args.save_path if args.save_path else out_dir / f"{stem}.pt"
    log_path = root / args.log_path if args.log_path else out_dir / f"{stem}_train_log.csv"
    fig_path = root / args.fig_path if args.fig_path else out_dir / f"{stem}_train_return.png"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path, log_path, fig_path


def save_training_curve(history_return: list[float], fig_path: Path,
                        title: str) -> None:
    """에피소드별 실제 누적 보상과 최근 50개 이동평균을 그린다."""
    window = max(1, min(50, len(history_return)))
    # 최초 49개 구간에서는 지금까지 기록된 보상만 평균한다.
    moving = [
        sum(history_return[max(0, i - window + 1) : i + 1]) / (i - max(0, i - window + 1) + 1)
        for i in range(len(history_return))
    ]
    plt.figure(figsize=(8, 4))
    plt.plot(history_return, color="lightgray", label="episode return")
    plt.plot(moving, color="steelblue", label=f"{window}-episode mean")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()


def finish_training(agent, env, history_return: list[float], save_path: Path,
                    log_path: Path, fig_path: Path, title: str) -> None:
    """학습된 모델을 저장하고 환경을 닫은 뒤 학습 곡선을 생성한다."""
    agent.save(save_path)
    env.close()
    save_training_curve(history_return, fig_path, title)
    print(f"저장: {save_path}")
    print(f"로그: {log_path}")
    print(f"곡선: {fig_path}")
