# 강화학습 재고 발주 프로젝트

`gym-invmgmt` 원본 환경으로 재고 발주 시뮬레이션을 돌릴 수 있는 1단계 구성입니다.  
지금은 학습 코드를 넣지 않았고, 원본 파일을 가져와 환경을 설치한 뒤 예제를 실행할 수 있게만 맞춰 두었습니다.

## 원본과 이 프로젝트에서 추가한 것

### 원본 (`vendor/gym-invmgmt`)

[r2barati/gym-invmgmt](https://github.com/r2barati/gym-invmgmt) 태그 `v0.2.1` (커밋 `e3e1e2132cfb6d3d7c2c8e7f2e79e670a994d97d`)에서 가져온 파일입니다. 가져온 원본 파일은 수정하지 않았습니다.

| 경로 | 역할 |
|---|---|
| `vendor/gym-invmgmt/gym_invmgmt/` | 환경 패키지. `CoreEnv`, 수요 엔진, 네트워크 토폴로지, wrapper |
| `vendor/gym-invmgmt/examples/quickstart.py` | 원본 최소 실행 예제. `GymInvMgmt/MultiEchelon-v0` 1에피소드 |
| `vendor/gym-invmgmt/examples/run_heuristic.py` | 원본 Newsvendor base-stock 휴리스틱 예제 |
| `vendor/gym-invmgmt/pyproject.toml` | 원본 패키지 버전 `0.2.1`과 의존성 정의 |

원본 패키지가 등록하는 환경 ID는 아래 두 개입니다. `vendor/gym-invmgmt/gym_invmgmt/__init__.py`에서 확인했습니다.

- `GymInvMgmt/MultiEchelon-v0` — 분기형 네트워크, 기본 30기간
- `GymInvMgmt/Serial-v0` — 직렬 공급망, 기본 30기간

### 이 프로젝트에서 추가한 것

| 경로 | 역할 |
|---|---|
| `.venv/` | 로컬 Python 가상환경. Git에 넣지 않음 |
| `requirements.txt` | 원본 패키지를 editable로 설치하고 의존성 버전을 고정 |
| `vendor/SOURCE.md` | 가져온 태그, 커밋, 제외한 원본 항목 |
| `.gitignore` | 가상환경과 캐시 제외 |
| `README.md` | 이 파일 |

원본에서 시뮬레이션에 필요 없어 가져오지 않은 항목은 `vendor/SOURCE.md`에 적어 두었습니다. CI, 문서, 테스트, 학습/벤치마크 예제는 포함하지 않았습니다.

## 버전 고정

- `gym-invmgmt==0.2.1` — PyPI가 아니라 `vendor/gym-invmgmt` 원본 소스를 설치합니다.
- 확인한 설치 결과: `gym_invmgmt.__version__ == "0.2.1"`

## 설치

Python 3.13.15, MacBook Pro M4에서 아래 명령으로 설치하고 실행을 확인했습니다.

```bash
cd inventory-rl-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

설치 후 확인한 패키지 버전:

- gym-invmgmt 0.2.1
- gymnasium 1.4.0
- numpy 2.5.4
- scipy 1.18.1
- networkx 3.7
- matplotlib 3.11.2

## 실행

가상환경을 켠 다음 원본 예제를 실행합니다.

```bash
source .venv/bin/activate
python vendor/gym-invmgmt/examples/quickstart.py
```

이 프로젝트에서 실행한 결과:

```text
Environment: GymInvMgmt/MultiEchelon-v0
  Observation shape: (71,)
  Action shape:      (11,)
  Num periods:       30

Episode finished in 30 steps
  Total reward:    -54,154.17
  Total inventory: 181076
  Total backlog:   0

KPIs:
  Fill rate:       100.00%
  Inventory turns: 0.04
  Avg inventory:   14686
```

직렬 기본 환경도 같은 원본 API로 1에피소드가 끝나는 것을 확인했습니다.

```python
import gymnasium as gym
import gym_invmgmt

env = gym.make("GymInvMgmt/Serial-v0")
obs, info = env.reset(seed=42)
```

확인한 `Serial-v0` 값:

- 관측 차원 15, 행동 차원 3, 기간 30
- 노드 `[0, 1, 2, 3, 4]`, 발주 링크 `[(2, 1), (3, 2), (4, 3)]`, 소매 링크 `[(1, 0)]`
- `step()`이 돌려준 `info` 키: `period`, `sentiment`, `step_reward`, `total_backlog`, `total_inventory`
- `seed=42`에서 무작위 행동 1에피소드 총보상 `-29027.63`

원본 휴리스틱 예제도 같은 방식으로 실행할 수 있습니다.

```bash
python vendor/gym-invmgmt/examples/run_heuristic.py
```

## 한 기간씩 살펴보기

원본 `Serial-v0`를 고정량 발주로 한 기간씩 출력하는 코드는 `scripts/walk_serial_steps.py`입니다. 이 파일은 우리가 추가한 설명용 스크립트입니다.

```bash
source .venv/bin/activate
python scripts/walk_serial_steps.py --steps 3 --order 20 --seed 42
```

`reset`과 `step(action)`의 원본 반환값을 노트북에서 보려면 `notebooks/inspect_env_interface.ipynb`를 열고 커널을 `.venv`로 고른 뒤 위에서부터 실행합니다. `ACTION` 숫자만 바꾸고 `step` 셀을 다시 실행하면 됩니다.
