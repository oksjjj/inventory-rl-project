"""고정 발주, 재고 수준, REINFORCE를 같은 KPI로 비교한다.

사용:
    python scripts/compare_policies.py
    python scripts/compare_policies.py --demand stationary
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_rl.env import demand_label, make_serial_env, output_dir_for_demand
from inventory_rl.evaluate import evaluate_policy
from inventory_rl.policies import BaseStockPolicy, ConstantOrderPolicy
from inventory_rl.reinforce_agent import ReinforceAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial-v0 정책 비교")
    parser.add_argument("--demand", choices=["stationary", "trend_shock"], default="trend_shock")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--constant-order", type=float, default=20.0)
    parser.add_argument("--out-csv", type=str, default="")
    return parser.parse_args()


def print_table(rows: list[dict]) -> None:
    """터미널에 계획서 KPI를 맞춰 출력한다."""
    headers = [
        ("name", "정책", 22),
        ("total_profit", "누적이익", 12),
        ("fill_rate", "충족률", 10),
        ("total_backlog", "미충족", 10),
        ("avg_inventory", "평균재고", 10),
        ("mean_sentiment", "sentiment", 10),
    ]
    line = " ".join(title.ljust(width) for _, title, width in headers)
    print(line)
    print("-" * len(line))
    for row in rows:
        cells = []
        for key, _, width in headers:
            value = row[key]
            if key == "name":
                text = str(value)
            elif key == "fill_rate":
                text = f"{value:.1%}"
            else:
                text = f"{value:,.2f}"
            cells.append(text.ljust(width))
        print(" ".join(cells))


def main() -> None:
    args = parse_args()
    out_dir = ROOT / (args.out_dir or output_dir_for_demand(args.demand))
    # 학습 때와 다른 시드(기본 1000부터)로 평가한다.
    seeds = list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
    env = make_serial_env(demand=args.demand)
    env.reset(seed=0)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    constant = ConstantOrderPolicy(order=args.constant_order)
    base_stock = BaseStockPolicy(env)

    policies = [
        ("고정 발주", lambda obs, current_env, policy=constant: policy.get_action(obs, current_env)),
        ("재고 수준", lambda obs, current_env, policy=base_stock: policy.get_action(obs, current_env)),
    ]

    checkpoint = ROOT / args.checkpoint if args.checkpoint else out_dir / "reinforce.pt"
    if checkpoint.exists():
        agent = ReinforceAgent(obs_dim=obs_dim, act_dim=act_dim)
        agent.load(checkpoint)
        # 평가에서는 평균 발주만 쓴다. 학습 때처럼 가우시안에서 뽑지 않는다.
        policies.append(
            (
                "학습 정책(REINFORCE)",
                lambda obs, current_env, learned=agent: learned.get_action(obs, current_env, deterministic=True),
            )
        )
    else:
        print(f"체크포인트가 없어 학습 정책은 비교에서 빠집니다: {checkpoint}")
        print(f"먼저 python scripts/train_reinforce.py --demand {args.demand} 를 실행하세요.")

    env.close()

    rows = []
    for name, action_fn in policies:
        print(f"평가 중: {name}")
        summary = evaluate_policy(action_fn, seeds, demand=args.demand)
        summary["name"] = name
        rows.append(summary)

    print()
    print(f"수요: {demand_label(args.demand)}")
    print(f"평가 에피소드 {args.eval_episodes}개, 시작 시드 {args.eval_seed}")
    print_table(rows)

    out_csv = ROOT / args.out_csv if args.out_csv else out_dir / "policy_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "total_profit",
        "total_profit_std",
        "fill_rate",
        "fill_rate_std",
        "total_backlog",
        "avg_inventory",
        "mean_sentiment",
        "episode_return",
        "n_episodes",
    ]
    with out_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장: {out_csv}")


if __name__ == "__main__":
    main()
