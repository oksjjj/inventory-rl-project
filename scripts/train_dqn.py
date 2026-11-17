"""이산 발주를 DQN으로 학습한다.

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_rl.dqn_agent import DQNAgent
from inventory_rl.env import clip_order, make_serial_env, output_dir_for_demand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial-v0 이산 발주 DQN 학습")
    parser.add_argument("--demand", choices=["stationary", "trend_shock"], default="trend_shock")
    parser.add_argument("--episodes", type=int, default=3000)
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
        obs_dim=env.observation_space.shape[0],
        n_links=env.action_space.shape[0],
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
                action, index = agent.act_and_index(obs)
                next_obs, reward, terminated, truncated, _ = env.step(clip_order(action))
                done = terminated or truncated
                agent.remember(obs, index, float(reward), next_obs, done)
                stats = agent.learn()
                last_loss = stats["loss"]
                rewards.append(float(reward))
                obs = next_obs

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
