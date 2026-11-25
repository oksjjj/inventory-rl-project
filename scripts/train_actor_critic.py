"""발주량 전체를 TD actor-critic으로 학습한다. (수업 8강)

REINFORCE와 같은 가우시안 정책이지만, 에피소드가 끝나기를 기다리지 않는다.
매 기간 δ = r + γV(s') − V(s) 로 actor와 critic을 바꾼다.

사용:
    python scripts/train_actor_critic.py
    python scripts/train_actor_critic.py --demand stationary
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


# scripts/ 에서 실행해도 src/inventory_rl 을 import 하게 경로를 넣는다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_rl.actor_critic_agent import ActorCriticAgent
from inventory_rl.env import clip_order, make_serial_env
from inventory_rl.training_utils import (
    parse_train_args, seed_training, training_paths, finish_training,
)


def main() -> None:
    args = parse_train_args("Serial-v0 전체 발주 TD actor-critic 학습")
    seed_training(args.seed)
    save_path, log_path, fig_path = training_paths(ROOT, args, "actor_critic")
    env = make_serial_env(demand=args.demand)
    env.reset(seed=args.seed)
    agent = ActorCriticAgent(
        obs_dim=env.observation_space.shape[0],  # Serial-v0 = 15
        act_dim=env.action_space.shape[0],  # 발주 링크 3개
    )

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
            # train() 은 act_and_log / learn_step 안에서 호출한다.
            while not done:
                action, log_prob, entropy, value = agent.act_and_log(obs)
                next_obs, reward, terminated, truncated, _ = env.step(clip_order(action))
                done = terminated or truncated
                # REINFORCE 와 달리 30기간을 기다리지 않고, 이 한 칸으로 바로 업데이트
                last_stats = agent.learn_step(
                    log_prob, entropy, value, float(reward), next_obs, done
                )
                rewards.append(float(reward))
                obs = next_obs  # 다음 루프의 s 가 지금 s'

            episode_return = float(sum(rewards))
            history_return.append(episode_return)
            writer.writerow({"episode": episode, "return": episode_return, **last_stats})
            if episode % 50 == 0 or episode == 1:
                print(
                    f"episode {episode:4d}  return={episode_return:8.2f}  "
                    f"loss={last_stats['loss']:.3f}  policy={last_stats['policy_loss']:.3f}  "
                    f"value={last_stats['value_loss']:.3f}"
                )

    finish_training(agent, env, history_return, save_path, log_path, fig_path,
                    f"TD actor-critic (full order), demand={args.demand}")


if __name__ == "__main__":
    main()
