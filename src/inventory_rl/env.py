"""Serial-v0 환경을 만드는 곳.

원본 패키지는 vendor/gym-invmgmt 이다. 여기서는 gym.make 인자만 고른다.
gym.make에 demand_config를 넘기면 원본 기본 수요 설정을 통째로 바꾼다.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import gym_invmgmt  # noqa: F401  # 이 import가 GymInvMgmt/Serial-v0 를 등록한다.

# 학습/비교에서 고를 수 있는 수요.
# stationary: 평균이 계속 20 근처. 설명용 walk_serial_steps 와 같다.
# trend_shock: 완만히 늘다가 기간 15에 평균이 2배. 과제 동기(수요 증가·급변).
DEMAND_PRESETS = {
    "stationary": {
        "type": "stationary",
        "base_mu": 20,
        "use_goodwill": True,  # 미충족이 있으면 다음 수요가 줄어든다.
    },
    "trend_shock": {
        "type": "trend+shock",
        "base_mu": 20,
        "use_goodwill": True,
        "trend_slope": 0.03,  # μ(t) *= 1 + 0.03 * t
        "shock_time": 15,  # 이 기간부터 평균에 shock_mag 를 곱한다.
        "shock_mag": 2.0,
    },
}

# 기본값은 급변 수요. 예전 코드가 SERIAL_DEMAND_CONFIG 를 직접 보던 호환용.
SERIAL_DEMAND_CONFIG = DEMAND_PRESETS["trend_shock"]

# 원본 action_space 상한은 용량×기간이라 학습에 쓰지 않는다. 발주는 0~80으로 자른다.
MAX_ORDER = 80.0


def demand_config(demand: str) -> dict:
    """이름(stationary / trend_shock)을 원본 demand_engine 설정으로 바꾼다."""
    if demand not in DEMAND_PRESETS:
        raise ValueError(f"unknown demand {demand!r}. use one of {sorted(DEMAND_PRESETS)}")
    return DEMAND_PRESETS[demand]


def demand_label(demand: str) -> str:
    """비교 표 위에 찍을 한글 설명."""
    if demand == "stationary":
        return "정상 Poisson(base_mu=20), goodwill 사용"
    return "완만히 증가하다 기간 15에 평균이 2배(trend+shock), goodwill 사용"


def output_dir_for_demand(demand: str) -> str:
    """수요마다 체크포인트가 섞이지 않게 저장 폴더를 나눈다."""
    if demand == "stationary":
        return "outputs/stationary"
    return "outputs"


def make_serial_env(num_periods: int = 30, demand: str = "trend_shock") -> gym.Env:
    """원본 Serial-v0 를 만든다.

    노드: 시장 0 ← 소매 1 ← 유통 2 ← 공장 3 ← 원자재 4
    행동 3칸: 링크 (2,1) L=4, (3,2) L=4, (4,3) L=0 에 넣을 발주량.
    관측 15칸: obs.py 주석 참고.
    """
    return gym.make(
        "GymInvMgmt/Serial-v0",
        demand_config=demand_config(demand),
        num_periods=num_periods,
    )


def clip_order(action: np.ndarray) -> np.ndarray:
    """음수 발주와 너무 큰 발주를 막는다."""
    return np.clip(np.asarray(action, dtype=np.float64), 0.0, MAX_ORDER)
