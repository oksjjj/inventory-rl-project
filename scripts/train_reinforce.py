"""발주량 전체를 순수 REINFORCE로 학습한다. (수업 8강)

에피소드가 끝난 뒤에 update_episode를 한 번 호출한다.

한 에피소드 흐름:
    reset → (관측 보고 발주 뽑기 → step → 보상 저장) × 30기간
         → 모은 보상으로 네트워크를 한 번 업데이트

사용:
    python scripts/train_reinforce.py
    python scripts/train_reinforce.py --demand stationary
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


# scripts/ 에서 실행해도 src/inventory_rl 을 import 하게 경로를 넣는다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_rl.env import clip_order, make_serial_env
from inventory_rl.training_utils import (
    parse_train_args, seed_training, training_paths, finish_training,
)
from inventory_rl.reinforce_agent import ReinforceAgent


def main() -> None:
    args = parse_train_args("Serial-v0 전체 발주 REINFORCE 학습")
    seed_training(args.seed)
    save_path, log_path, fig_path = training_paths(ROOT, args, "reinforce_vanilla")
    env = make_serial_env(demand=args.demand)
    env.reset(seed=args.seed)
    agent = ReinforceAgent(
        obs_dim=env.observation_space.shape[0],  # Serial-v0 = 15
        act_dim=env.action_space.shape[0],  # 발주 링크 3개
    )

    history_return = []

    with log_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "return", "loss"])
        writer.writeheader()

        for episode in range(1, args.episodes + 1):
            # 에피소드마다 시드를 바꿔 수요 난수를 다르게 한다. 예: seed=0이면 1, 2, ..., 3000
            obs, _ = env.reset(seed=args.seed + episode)
            log_probs = []
            rewards = []
            done = False
            # train() 은 act_and_log / update_episode 안에서 호출한다.
            # 한 에피소드 = 기본 30기간. 여기서는 아직 가중치를 안 바꾼다.
            while not done:
                action, log_prob = agent.act_and_log(obs)
                obs, reward, terminated, truncated, _ = env.step(clip_order(action))
                done = terminated or truncated  # 30기간이 끝나면 truncated=True
                log_probs.append(log_prob)  # 이 발주의 로그확률 (업데이트에 필요)
                rewards.append(float(reward))  # 원본 한 기간 이익

            # 30기간이 다 끝난 뒤에야 G_t 를 만들 수 있다. 그래서 지금 업데이트.
            stats = agent.update_episode(log_probs, rewards)
            episode_return = stats["return"]
            history_return.append(episode_return)
            writer.writerow({"episode": episode, **stats})
            if episode % 50 == 0 or episode == 1:
                print(
                    f"episode {episode:4d}  return={episode_return:8.2f}  "
                    f"loss={stats['loss']:.3f}"
                )

    finish_training(agent, env, history_return, save_path, log_path, fig_path,
                    f"Vanilla REINFORCE (full order), demand={args.demand}")


if __name__ == "__main__":
    main()
