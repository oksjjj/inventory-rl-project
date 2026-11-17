"""발주량 전체를 TD actor-critic으로 학습한다. (수업 8강)

예전에는 이 파일이 train_reinforce.py 의 별칭이었다. 지금은 별도 알고리즘이다.
REINFORCE와 같은 가우시안 정책이지만, 에피소드가 끝나기를 기다리지 않는다.
매 기간 δ = r + γV(s') − V(s) 로 actor와 critic을 바꾼다.

사용:
    python scripts/train_actor_critic.py
    python scripts/train_actor_critic.py --demand stationary
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_rl.actor_critic_agent import ActorCriticAgent
from inventory_rl.env import clip_order, make_serial_env, output_dir_for_demand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial-v0 전체 발주 TD actor-critic 학습")
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
    save_path = ROOT / args.save_path if args.save_path else out_dir / "actor_critic.pt"
    fig_path = ROOT / args.fig_path if args.fig_path else out_dir / "actor_critic_train_return.png"
    env = make_serial_env(demand=args.demand)
    env.reset(seed=args.seed)
    agent = ActorCriticAgent(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
    )

    log_path = ROOT / args.log_path if args.log_path else out_dir / "actor_critic_train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history_return = []

    with log_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["episode", "return", "loss", "policy_loss", "value_loss", "td_error"],
        )
        writer.writeheader()

        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset(seed=args.seed + episode)
            rewards = []
            last_stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "td_error": 0.0}
            done = False
            agent.model.train()
            while not done:
                action, log_prob, entropy, value = agent.act_and_log(obs)
                next_obs, reward, terminated, truncated, _ = env.step(clip_order(action))
                done = terminated or truncated
                last_stats = agent.learn_step(
                    log_prob, entropy, value, float(reward), next_obs, done
                )
                rewards.append(float(reward))
                obs = next_obs

            episode_return = float(sum(rewards))
            history_return.append(episode_return)
            writer.writerow({"episode": episode, "return": episode_return, **last_stats})
            if episode % 50 == 0 or episode == 1:
                print(
                    f"episode {episode:4d}  return={episode_return:8.2f}  "
                    f"loss={last_stats['loss']:.3f}  policy={last_stats['policy_loss']:.3f}  "
                    f"value={last_stats['value_loss']:.3f}"
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
    plt.title(f"TD actor-critic (full order), demand={args.demand}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()

    print(f"저장: {save_path}")
    print(f"로그: {log_path}")
    print(f"곡선: {fig_path}")


if __name__ == "__main__":
    main()
