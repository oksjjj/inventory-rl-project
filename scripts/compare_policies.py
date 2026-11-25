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
from inventory_rl.discrete_agent import DiscreteReinforceAgent, DiscreteActorCriticAgent
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
    parser.add_argument("--discrete-rf-checkpoint", type=str, default="")
    parser.add_argument("--discrete-ac-checkpoint", type=str, default="")
    parser.add_argument("--require-all", action="store_true", help="모델이 누락되면 비교 중단")
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

    checkpoint = ROOT / args.checkpoint if args.checkpoint else out_dir / "reinforce_vanilla.pt"
    if checkpoint.exists():
        agent = ReinforceAgent(obs_dim=obs_dim, act_dim=act_dim)
        agent.load(checkpoint)
        # 평가에서는 평균 발주만 쓴다. 학습 때처럼 가우시안에서 뽑지 않는다.
        policies.append(
            (
                "REINFORCE (Continuous)",
                lambda obs, current_env, learned=agent: learned.get_action(obs, current_env, deterministic=True),
            )
        )
    else:
        print(f"체크포인트가 없어 학습 정책은 비교에서 빠집니다: {checkpoint}")
        print(f"먼저 python scripts/train_reinforce.py --demand {args.demand} 를 실행하세요.")

    ac_ckpt = ROOT / args.ac_checkpoint if args.ac_checkpoint else out_dir / "actor_critic.pt"
    if ac_ckpt.exists():
        ac_agent = ActorCriticAgent(obs_dim=obs_dim, act_dim=act_dim)
        ac_agent.load(ac_ckpt)
        policies.append(
            (
                "Actor-Critic (Continuous)",
                lambda obs, current_env, learned=ac_agent: learned.get_action(
                    obs, current_env, deterministic=True
                ),
            )
        )
    else:
        print(f"체크포인트가 없어 actor-critic은 비교에서 빠집니다: {ac_ckpt}")
        print(f"먼저 python scripts/train_actor_critic.py --demand {args.demand} 를 실행하세요.")

    dqn_ckpt = ROOT / args.dqn_checkpoint if args.dqn_checkpoint else out_dir / "dqn.pt"
    if dqn_ckpt.exists():
        dqn = DQNAgent(obs_dim=obs_dim, n_links=act_dim)
        dqn.load(dqn_ckpt)
        policies.append(
            (
                "DQN (Discrete)",
                lambda obs, current_env, learned=dqn: learned.get_action(
                    obs, current_env, deterministic=True
                ),
            )
        )
    else:
        print(f"체크포인트가 없어 DQN은 비교에서 빠집니다: {dqn_ckpt}")
        print(f"먼저 python scripts/train_dqn.py --demand {args.demand} 를 실행하세요.")

    # Categorical 정책은 평가 시 최대 확률의 행동을 선택한다.
    discrete_specs = [
        ("REINFORCE (Discrete)", DiscreteReinforceAgent,
         ROOT / args.discrete_rf_checkpoint if args.discrete_rf_checkpoint else out_dir / "reinforce_discrete.pt",
         "train_reinforce.py"),
        ("Actor-Critic (Discrete)", DiscreteActorCriticAgent,
         ROOT / args.discrete_ac_checkpoint if args.discrete_ac_checkpoint else out_dir / "actor_critic_discrete.pt",
         "train_actor_critic.py"),
    ]
    for name, agent_cls, ckpt, script in discrete_specs:
        if ckpt.exists():
            discrete_agent = agent_cls(obs_dim=obs_dim, act_dim=act_dim)
            discrete_agent.load(ckpt)
            policies.append(
                (name, lambda obs, current_env, learned=discrete_agent:
                 learned.get_action(obs, current_env, deterministic=True))
            )
        else:
            print(f"체크포인트가 없어 {name}은 비교에서 빠집니다: {ckpt}")
            print(f"먼저 python scripts/{script} --action-space discrete --demand {args.demand} 를 실행하세요.")

    env.close()
    if args.require_all and len(policies) != 7:
        raise SystemExit(f"7개 정책이 필요하지만 {len(policies)}개만 준비되었습니다. 누락 모델을 먼저 학습하세요.")

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

    out_csv = ROOT / args.out_csv if args.out_csv else out_dir / "policy_comparison_7.csv"
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
