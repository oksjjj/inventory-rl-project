"""이산 발주를 DQN으로 학습한다. (수업 7강)

원본 발주는 연속이다. dqn_agent.ORDER_LEVELS 로 칸을 나눈 뒤 Q 를 학습한다.

사용:
    python scripts/train_dqn.py
    python scripts/train_dqn.py --demand stationary
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


# scripts/ 에서 실행해도 src/inventory_rl 을 import 하게 경로를 넣는다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_rl.dqn_agent import DQNAgent
from inventory_rl.env import clip_order, make_serial_env
from inventory_rl.training_utils import (
    parse_train_args, seed_training, training_paths, finish_training,
)


def main() -> None:
    args = parse_train_args("Serial-v0 이산 발주 DQN 학습")
    seed_training(args.seed)
    save_path, log_path, fig_path = training_paths(ROOT, args, "dqn")
    env = make_serial_env(demand=args.demand)
    env.reset(seed=args.seed)
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],  # Serial-v0 = 15
        n_links=env.action_space.shape[0],  # 발주 링크 3개 → 이산 행동 125개
    )

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

    finish_training(agent, env, history_return, save_path, log_path, fig_path,
                    f"DQN (discrete orders), demand={args.demand}")


if __name__ == "__main__":
    main()
