"""고정 발주, 재고 수준, REINFORCE, actor-critic, DQN을 같은 KPI로 비교한다.

사용:
    python scripts/compare_policies.py
    python scripts/compare_policies.py --demand stationary
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# scripts/ 에서 실행해도 src/inventory_rl 을 import 하게 경로를 넣는다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_rl.actor_critic_agent import ActorCriticAgent
from inventory_rl.dqn_agent import DQNAgent
from inventory_rl.env import demand_label, make_serial_env, output_dir_for_demand
from inventory_rl.evaluate import evaluate_policy
from inventory_rl.policies import BaseStockPolicy, ConstantOrderPolicy
from inventory_rl.reinforce_agent import ReinforceAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial-v0 정책 비교")
    parser.add_argument("--demand", choices=["stationary", "trend_shock"], default="trend_shock")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--dqn-checkpoint", type=str, default="")
    parser.add_argument("--ac-checkpoint", type=str, default="")
    parser.add_argument("--require-all", action="store_true", help="5개 정책 중 체크포인트가 누락되면 비교 중단")
    parser.add_argument("--eval-episodes", type=int, default=20)  # 학습이 아니라 평균 낼 판 수
    parser.add_argument("--eval-seed", type=int, default=10000)  # 기본 학습 환경 시드 1~3000과 겹치지 않게
    parser.add_argument("--constant-order", type=float, default=20.0)  # 고정 발주 정책의 수량
    parser.add_argument("--out-csv", type=str, default="")
    return parser.parse_args()


def print_table(rows: list[dict]) -> None:
    """터미널에 계획서 KPI를 맞춰 출력한다."""
    # (딕셔너리 키, 화면에 찍을 제목, 칸 너비)
    headers = [
        ("name", "정책", 26),
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
                text = f"{value:.1%}"  # 예: 0.932 → 93.2%
            else:
                text = f"{value:,.2f}"
            cells.append(text.ljust(width))
        print(" ".join(cells))


def main() -> None:
    args = parse_args()
    out_dir = ROOT / (args.out_dir or output_dir_for_demand(args.demand))
    # 기본 학습 환경 시드와 겹치지 않는 평가 시드 10000..10019.
    seeds = list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
    env = make_serial_env(demand=args.demand)
    env.reset(seed=0)
    obs_dim = env.observation_space.shape[0]  # 15
    act_dim = env.action_space.shape[0]  # 3

    constant = ConstantOrderPolicy(order=args.constant_order)
    base_stock = BaseStockPolicy(env)  # 노드별 목표 S 를 여기서 미리 계산

    policies = [
        ("고정 발주", lambda obs, current_env, policy=constant: policy.get_action(obs, current_env)),
        ("재고 수준", lambda obs, current_env, policy=base_stock: policy.get_action(obs, current_env)),
    ]

    # 모델 종류에 따라 생성 인자만 다르고, 체크포인트 로드와 평가 등록은 동일하다.
    learned_specs = [
        ("학습 정책(Vanilla REINFORCE)", ReinforceAgent,
         {"obs_dim": obs_dim, "act_dim": act_dim},
         args.checkpoint, "reinforce_vanilla.pt", "train_reinforce.py"),
        ("학습 정책(Actor-Critic)", ActorCriticAgent,
         {"obs_dim": obs_dim, "act_dim": act_dim},
         args.ac_checkpoint, "actor_critic.pt", "train_actor_critic.py"),
        ("학습 정책(DQN)", DQNAgent,
         {"obs_dim": obs_dim, "n_links": act_dim},
         args.dqn_checkpoint, "dqn.pt", "train_dqn.py"),
    ]
    for name, agent_cls, kwargs, custom_path, filename, script in learned_specs:
        checkpoint = ROOT / custom_path if custom_path else out_dir / filename
        if not checkpoint.exists():
            print(f"체크포인트가 없어 {name}은 비교에서 빠집니다: {checkpoint}")
            print(f"먼저 python scripts/{script} --demand {args.demand} 를 실행하세요.")
            continue
        agent = agent_cls(**kwargs)
        agent.load(checkpoint)
        # 평가 시 Gaussian은 평균 발주량, DQN은 최대 Q 행동을 선택한다.
        policies.append(
            (name, lambda obs, current_env, learned=agent:
             learned.get_action(obs, current_env, deterministic=True))
        )

    env.close()
    if args.require_all and len(policies) != 5:
        raise SystemExit(f"5개 정책이 필요하지만 {len(policies)}개만 준비되었습니다. 누락 모델을 먼저 학습하세요.")

    rows = []
    for name, action_fn in policies:
        print(f"평가 중: {name}")
        summary = evaluate_policy(action_fn, seeds, demand=args.demand)  # 20에피소드 평균 KPI
        summary["name"] = name
        rows.append(summary)

    print()
    print(f"수요: {demand_label(args.demand)}")
    print(f"평가 에피소드 {args.eval_episodes}개, 시작 시드 {args.eval_seed}")
    print_table(rows)

    out_csv = ROOT / args.out_csv if args.out_csv else out_dir / "policy_comparison_vanilla.csv"
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
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")  # 표에 없는 키는 버림
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장: {out_csv}")


if __name__ == "__main__":
    main()
