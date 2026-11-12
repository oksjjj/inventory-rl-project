"""발주량 전체를 REINFORCE + 기준선으로 학습한다.

한 에피소드 흐름:
    reset → (관측 보고 발주 뽑기 → step → 보상 저장) × 30기간
         → 모은 보상으로 네트워크를 한 번 업데이트

사용:
    python scripts/train_reinforce.py
    python scripts/train_reinforce.py --demand stationary
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_rl.env import clip_order, make_serial_env, output_dir_for_demand
from inventory_rl.reinforce_agent import ReinforceAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial-v0 전체 발주 REINFORCE 학습")
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
    # 정상 수요와 급변 수요 가중치가 서로 덮이지 않게 폴더를 나눈다.
    out_dir = ROOT / (args.out_dir or output_dir_for_demand(args.demand))
    save_path = ROOT / args.save_path if args.save_path else out_dir / "reinforce.pt"
    fig_path = ROOT / args.fig_path if args.fig_path else out_dir / "reinforce_train_return.png"
    env = make_serial_env(demand=args.demand)
    env.reset(seed=args.seed)
    agent = ReinforceAgent(
        obs_dim=env.observation_space.shape[0],  # Serial-v0 = 15
        act_dim=env.action_space.shape[0],  # 발주 링크 3개
    )

    log_path = ROOT / args.log_path if args.log_path else out_dir / "reinforce_train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history_return = []

    with log_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "return", "loss", "policy_loss", "value_loss"])
        writer.writeheader()

        for episode in range(1, args.episodes + 1):
            # 에피소드마다 시드를 바꿔 수요 난수를 다르게 한다.
            obs, _ = env.reset(seed=args.seed + episode)
            log_probs = []
            values = []
            rewards = []
            entropies = []
            done = False
            agent.model.train()
            # 한 에피소드 = 기본 30기간. 여기서는 아직 가중치를 안 바꾼다.
            while not done:
                action, log_prob, entropy, value = agent.act_and_log(obs)
                obs, reward, terminated, truncated, _ = env.step(clip_order(action))
                done = terminated or truncated
                log_probs.append(log_prob)  # 이 발주의 로그확률 (업데이트에 필요)
                values.append(value)  # 그때의 V(s)
                entropies.append(entropy)
                rewards.append(float(reward))  # 원본 한 기간 이익

            # 30기간이 다 끝난 뒤에야 G_t 를 만들 수 있다. 그래서 지금 업데이트.
            stats = agent.update_episode(log_probs, values, rewards, entropies)
            episode_return = float(sum(rewards))
            history_return.append(episode_return)
            writer.writerow({"episode": episode, **stats})
            if episode % 50 == 0 or episode == 1:
                print(
                    f"episode {episode:4d}  return={episode_return:8.2f}  "
                    f"loss={stats['loss']:.3f}  policy={stats['policy_loss']:.3f}  "
                    f"value={stats['value_loss']:.3f}"
                )

    agent.save(save_path)
    env.close()

    # 한 에피소드 보상은 수요 난수 때문에 흔들린다. 50개 평균으로 추세를 본다.
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
    plt.title(f"REINFORCE + baseline (full order), demand={args.demand}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()

    print(f"저장: {save_path}")
    print(f"로그: {log_path}")
    print(f"곡선: {fig_path}")


if __name__ == "__main__":
    main()
