# 강화학습 재고 발주 프로젝트

`gym-invmgmt` 원본 환경에서 Serial-v0로 재고 발주를 학습하고, 고정 발주·재고 수준 정책과 같은 지표로 비교합니다.

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
| `src/inventory_rl/` | 학습·기준선·평가 코드 |
| `scripts/train_reinforce.py` | REINFORCE 학습 |
| `scripts/compare_policies.py` | 고정 발주·재고 수준·학습 정책 비교 |
| `docs/rl_comparison.md` | 방법과 비교 결과 설명 |
| `docs/file_structure.md` | 파일 사이 호출 관계 |
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

## 학습과 비교

학습 정책은 순수 REINFORCE, TD Actor-Critic, DQN 세 가지입니다. 고정 발주·재고 수준과 같은 지표로 비교합니다. 순수 REINFORCE는 가치 함수와 엔트로피 보너스 없이 실제 할인 누적 보상으로 정책만 학습합니다. 자세한 설명은 [`docs/rl_comparison.md`](docs/rl_comparison.md)에 있습니다.

```bash
source .venv/bin/activate
python -m compileall src scripts

for demand in stationary trend_shock; do
    python scripts/train_dqn.py --demand "$demand" --episodes 3000 --seed 0 || break
    python scripts/train_reinforce.py --demand "$demand" --episodes 3000 --seed 0 || break
    python scripts/train_actor_critic.py --demand "$demand" --episodes 3000 --seed 0 || break
    python scripts/compare_policies.py --demand "$demand" --eval-episodes 20 --eval-seed 10000 --require-all || break
done
```

정상 수요로 보려면 `--demand stationary`를 쓰면 됩니다. 비교 지표는 누적 이익, 충족률, 미충족, 평균 재고, sentiment입니다.

새 모델은 `reinforce_vanilla.pt`, 새 비교는 `policy_comparison_vanilla.csv`로 저장합니다. 정상 수요는 `outputs/stationary/` 아래에 저장됩니다. 기존 모델과 CSV를 보존합니다.

최신 결과는 학습 3,000 에피소드, 평가 20 에피소드(환경 시드 10000~10019), 결정론적 정책 기준이다. 기존 7개 비교 CSV에서 최종 5개 정책 행만 추출한 값이며 새 학습 결과는 아니다. 이전 CSV는 이력으로 보존한다.

### 정상 수요

| 정책 | 누적 이익 | 충족률 | 잔여 미충족 | 평균 재고 | Goodwill |
|---|---:|---:|---:|---:|---:|
| 고정 발주 | 738.80 | 99.78% | 1.40 | 81.93 | 0.99 |
| 재고 수준 | 789.38 | 99.31% | 3.25 | 30.36 | 0.82 |
| 학습 정책(Vanilla REINFORCE) | 643.34 | 99.94% | 0.35 | 72.33 | 0.84 |
| 학습 정책(Actor-Critic) | 942.30 | 99.91% | 0.68 | 39.56 | 1.15 |
| 학습 정책(DQN) | 1088.53 | 99.21% | 5.60 | 37.78 | 1.15 |

### 급변 수요

| 정책 | 누적 이익 | 충족률 | 잔여 미충족 | 평균 재고 | Goodwill |
|---|---:|---:|---:|---:|---:|
| 고정 발주 | 726.41 | 94.80% | 35.50 | 77.62 | 0.63 |
| 재고 수준 | 816.92 | 89.96% | 64.65 | 26.28 | 0.61 |
| 학습 정책(Vanilla REINFORCE) | 733.54 | 95.54% | 29.36 | 59.91 | 0.60 |
| 학습 정책(Actor-Critic) | 1439.03 | 100.00% | 0.00 | 166.06 | 1.10 |
| 학습 정책(DQN) | 2033.90 | 93.22% | 102.95 | 63.27 | 1.09 |

DQN이 두 시나리오에서 가장 높은 누적 이익을 달성했다. 급변 수요에서 Actor-Critic은 충족률 100%지만 평균 재고가 166.06으로 높다. DQN의 경험 재사용과 고정 목표망, TD 학습 및 갱신 횟수 차이가 성능에 기여했을 가능성이 있으나 제거 실험 및 복수 학습 시드가 필요하다. REINFORCE는 순수 Monte Carlo 정책 경사이며 가치 기준선과 entropy 보너스를 쓰지 않는다.
