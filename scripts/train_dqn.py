"""이산 발주를 DQN으로 학습한다. (수업 7강)

원본 발주는 연속이다. dqn_agent.ORDER_LEVELS 로 칸을 나눈 뒤 Q 를 학습한다.

사용:
    python scripts/train_dqn.py
    python scripts/train_dqn.py --demand stationary
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# scripts/ 에서 실행해도 src/inventory_rl 을 import 하게 경로를 넣는다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_rl.dqn_agent import DQNAgent
from inventory_rl.env import clip_order, make_serial_env, output_dir_for_demand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial-v0 이산 발주 DQN 학습")
    parser.add_argument("--demand", choices=["stationary", "trend_shock"], default="trend_shock")
    parser.add_argument("--episodes", type=int, default=3000)  # 30기간짜리 게임을 몇 판 돌릴지
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--log-path", type=str, default="")
    parser.add_argument("--fig-path", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / (args.out_dir or output_dir_for_demand(args.demand))
    save_path = ROOT / args.save_path if args.save_path else out_dir / "dqn.pt"
    fig_path = ROOT / args.fig_path if args.fig_path else out_dir / "dqn_train_return.png"
    env = make_serial_env(demand=args.demand)
    env.reset(seed=args.seed)
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],  # Serial-v0 = 15
        n_links=env.action_space.shape[0],  # 발주 링크 3개 → 이산 행동 125개
    )

    log_path = ROOT / args.log_path if args.log_path else out_dir / "dqn_train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history_return = []
    last_loss = 0.0

    with log_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "return", "loss", "epsilon"])
        writer.writeheader()

        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset(seed=args.seed + episode)
            rewards = []
            done = False
            while not done:
                action, index = agent.act_and_index(obs)  # 예: index=6 → action=[20, 20, 0]
                next_obs, reward, terminated, truncated, _ = env.step(clip_order(action))
                done = terminated or truncated
                # 전이 (s, a, r, s') 를 버퍼에 넣고, 버퍼가 찼으면 Q 를 한 번 맞춤
                agent.remember(obs, index, float(reward), next_obs, done)
                stats = agent.learn()
                last_loss = stats["loss"]
                rewards.append(float(reward))
                obs = next_obs  # 다음 루프의 s 가 지금 s'

            episode_return = float(sum(rewards))
            history_return.append(episode_return)
            writer.writerow(
                {
                    "episode": episode,
                    "return": episode_return,
                    "loss": last_loss,
                    "epsilon": agent.epsilon(),
                }
            )
            if episode % 50 == 0 or episode == 1:
                print(
                    f"episode {episode:4d}  return={episode_return:8.2f}  "
                    f"loss={last_loss:.3f}  eps={agent.epsilon():.3f}"
                )

    agent.save(save_path)
    env.close()

    window = max(1, min(50, len(history_return)))
    # i=0 이면 자기 자신만, i=49 이면 0~49의 평균
    moving = [
        sum(history_return[max(0, i - window + 1) : i + 1]) / (i - max(0, i - window + 1) + 1)
        for i in range(len(history_return))
    ]
    plt.figure(figsize=(8, 4))
    plt.plot(history_return, color="lightgray", label="episode return")
    plt.plot(moving, color="steelblue", label=f"{window}-episode mean")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.title(f"DQN (discrete orders), demand={args.demand}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()

    print(f"저장: {save_path}")
    print(f"로그: {log_path}")
    print(f"곡선: {fig_path}")


if __name__ == "__main__":
    main()
