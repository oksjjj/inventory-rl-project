# 재고 발주 강화학습 (REINFORCE)

이 문서는 수업 과제용으로, 원본 시뮬레이터와 우리가 추가한 학습 코드를 구분해 적는다.  
학습 정책은 REINFORCE + 가치 기준선 하나다. 표 Q-learning은 쓰지 않는다.

## 1. 무엇을 푸는가

매 기간 세 발주 링크의 수량을 정한다. 목표는 에피소드 누적 이익을 키우는 것이다.

비교 정책은 세 개다.

| 정책 | 학습 | 하는 일 |
|---|---|---|
| 고정 발주 | 없음 | 매 기간 같은 수량 20 |
| 재고 수준 (base-stock) | 없음 | 재고 위치가 목표 S보다 모자라면 그 차이만큼 발주 |
| 학습 정책 | REINFORCE + 기준선 | 관측을 보고 발주량 전체를 학습 |

## 2. 원본과 우리가 추가한 것

원본은 `vendor/gym-invmgmt` (태그 `v0.2.1`)이다. 가져온 원본 파일은 수정하지 않았다.

| 경로 | 역할 |
|---|---|
| `src/inventory_rl/env.py` | Serial-v0 생성 |
| `src/inventory_rl/policies.py` | 고정 발주, 재고 수준 기준선 |
| `src/inventory_rl/reinforce_agent.py` | 가우시안 정책과 REINFORCE 업데이트 |
| `scripts/train_reinforce.py` | 학습 |
| `scripts/compare_policies.py` | 세 정책을 같은 KPI로 비교 |
| `scripts/walk_serial_steps.py` | 환경 설명용. 정상 수요를 그대로 둔다 |
| `notebooks/inspect_env_interface.ipynb` | `reset`/`step` 인터페이스 확인 |

## 3. 환경: Serial-v0

원본 ID는 `GymInvMgmt/Serial-v0`이다. 확인한 값:

- 기간 30, 관측 차원 15, 행동 차원 3
- 노드: 시장 0, 소매 1, 유통 2, 공장 3, 원자재 4
- 발주 링크: `(2,1)` lead 4, `(3,2)` lead 4, `(4,3)` lead 0
- 관측: 직전 수요, 미충족, 재고 3칸, 파이프라인 8칸, 기간 t, sentiment
- `step()`의 `info`: `period`, `sentiment`, `step_reward`, `total_backlog`, `total_inventory`

학습/비교 수요는 두 가지다.

- `stationary`: 정상 Poisson, base_mu=20, goodwill
- `trend_shock`: 완만히 증가하다 기간 15에 평균이 2배, goodwill

과제 동기(수요 증가·급변)에 맞는 기본값은 `trend_shock`이다. 대략 μ(0)=20, μ(14)=28.4, μ(15)=58, μ(29)=74.8 이다. goodwill이 켜져 있으면 이 값에 sentiment가 한 번 더 곱해진다.

원본 `action_space` 상한은 이론적 최댓값이어서 학습에 쓰지 않는다. 발주는 `[0, 80]`으로 자른다.

## 4. 보상과 goodwill

한 기간 보상은 원본 `core_env.py`의 `_STEP`과 같다.

```text
r = α^t (SR − PC − OC − HC − UP − FK)
```

| 기호 | 의미 |
|---|---|
| SR | 판매 수익 |
| PC | 구매 비용 |
| OC | 생산 비용 |
| HC | 재고 유지비 (보유 + 이동 중) |
| UP | 미충족 벌점 |
| FK | 고정 발주비 (이 토폴로지에서는 0) |
| α | 할인. Serial-v0 기본값 1 |

goodwill(sentiment)은 보상 항이 아니다. 직전 기간에 미충족이 있으면 수요 평균에 곱하는 배율이 줄어든다.

## 5. 왜 표 Q-learning이 아닌가

발주량은 연속이다. 세 링크를 이산 칸으로 자르면 조합이 바로 커진다. 수업의 함수 근사 + policy gradient로 가우시안 정책을 학습한다.

```text
발주 = clip(sigmoid(신경망) × 80, 0, 80)
```

에피소드가 끝나면 할인 보상 G_t 에서 가치 함수 기준선을 빼고, 그 값으로 로그확률을 올린다. 재고 수준 기준선은 시뮬레이터의 진짜 μ(t)를 보지 않고, 관측의 직전 수요만 쓴다.

## 6. 실행

```bash
source .venv/bin/activate

python scripts/train_reinforce.py --demand trend_shock --episodes 3000
python scripts/compare_policies.py --demand trend_shock

python scripts/train_reinforce.py --demand stationary --episodes 3000
python scripts/compare_policies.py --demand stationary
```

비교 지표는 계획서와 같다. 누적 이익, 충족률, 미충족, 평균 재고, sentiment.

- 급변 수요: `outputs/policy_comparison.csv`
- 정상 수요: `outputs/stationary/policy_comparison.csv`

## 7. 비교 결과

평가 에피소드 20개, 시드 1000부터. 학습 3000 에피소드.

### 급변 수요 (`trend+shock` + goodwill)

| 정책 | 누적 이익 | 충족률 | 미충족 | 평균 재고 | sentiment |
|---|---|---|---|---|---|
| 고정 발주 20 | 724.46 | 95.5% | 33.35 | 77.98 | 0.62 |
| 재고 수준 | 805.10 | 89.8% | 64.80 | 26.93 | 0.60 |
| 학습 정책 (REINFORCE) | 1659.64 | 94.4% | 82.21 | 117.97 | 1.06 |

고정 발주 20은 수요가 커진 뒤에 따라가지 못한다. 재고 수준은 직전 수요만 보므로 급변 뒤에 목표가 낮고, 재고를 낮게 가져 미충족이 커진다. 학습 정책은 재고를 더 들고 sentiment를 지켜 판매가 늘어 이익이 가장 크다.

### 정상 수요 (`stationary` + goodwill)

| 정책 | 누적 이익 | 충족률 | 미충족 | 평균 재고 | sentiment |
|---|---|---|---|---|---|
| 고정 발주 20 | 714.48 | 99.8% | 1.25 | 83.35 | 0.97 |
| 재고 수준 | 761.02 | 99.1% | 4.35 | 31.34 | 0.78 |
| 학습 정책 (REINFORCE) | 934.90 | 99.9% | 0.90 | 58.60 | 1.10 |

정상 수요에서도 학습 정책이 기준선을 넘는다. 재고 수준보다 재고는 많지만 sentiment가 높아 수요가 줄지 않는다.

## 8. 모델-free인지

학습 코드는 수요 식, 리드타임 식, 보상 식을 풀어 발주량을 계산하지 않는다. `reset`/`step`이 준 관측과 보상만 보고 정책을 바꾼다. 원본 환경 안에서 돌아가는 모델-free 학습이다.
