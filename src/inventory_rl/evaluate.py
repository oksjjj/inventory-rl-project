"""에피소드를 돌리고 계획서 KPI를 모은다.

KPI는 원본 gym_invmgmt.utils.compute_kpis 를 쓴다.
누적 이익, 충족률, 미충족, 평균 재고가 그 함수에서 나온다.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from gym_invmgmt.utils import compute_kpis

from inventory_rl.env import clip_order, make_serial_env

# 관측과 환경을 받아 발주 벡터를 돌려주는 함수.
ActionFn = Callable[[np.ndarray, object], np.ndarray]


def run_episode(env, action_fn: ActionFn, seed: int) -> dict[str, float]:
    """시드 하나로 30기간을 끝까지 돌린다. 학습 때와 달리 가중치는 안 바꾼다."""
    obs, _ = env.reset(seed=seed)
    rewards = []
    sentiments = []
    done = False
    while not done:
        action = clip_order(action_fn(obs, env))
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        sentiments.append(float(info.get("sentiment", 0.0)))
        done = terminated or truncated  # Serial-v0 는 기간이 끝나면 truncated=True

    # compute_kpis: 누적 이익, 충족률, 미충족 합, 평균 재고 (원본 함수).
    kpis = compute_kpis(env)
    kpis["episode_return"] = float(np.sum(rewards))
    kpis["mean_sentiment"] = float(np.mean(sentiments)) if sentiments else 1.0
    kpis["final_sentiment"] = float(env.unwrapped.demand_engine.sentiment)
    return kpis


def evaluate_policy(
    action_fn: ActionFn,
    seeds: list[int],
    num_periods: int = 30,
    demand: str = "trend_shock",
) -> dict[str, float]:
    """여러 시드의 평균과, 이익·충족률의 표준편차를 낸다."""
    env = make_serial_env(num_periods=num_periods, demand=demand)
    rows = [run_episode(env, action_fn, seed) for seed in seeds]
    keys = [
        "total_profit",
        "fill_rate",
        "total_backlog",
        "avg_inventory",
        "episode_return",
        "mean_sentiment",
    ]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary["fill_rate_std"] = float(np.std([row["fill_rate"] for row in rows]))
    summary["total_profit_std"] = float(np.std([row["total_profit"] for row in rows]))
    summary["n_episodes"] = float(len(seeds))
    env.close()
    return summary
