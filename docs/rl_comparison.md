# 재고 발주 강화학습

이 문서는 수업 과제용으로, 원본 시뮬레이터와 우리가 추가한 학습 코드를 구분해 적는다.

학습 정책은 세 개다. 파일·호출 관계와 수업 식 대응은 [`docs/file_structure.md`](file_structure.md) 를 본다.

- DQN: 7강, 이산 발주 `{0,20,40,60,80}`
- 순수 REINFORCE (가치 기준선 및 entropy 없음): 8강, 연속 발주, 에피소드 수익 $G_t$
- TD actor-critic: 8강, 연속 발주, 매 기간 $\delta_t$

현재 비교는 `scripts/compare_policies.py` 가 다섯 줄(고정·재고 수준·REINFORCE·actor-critic·DQN)을 출력한다.

## 1. 무엇을 푸는가

매 기간 세 발주 링크의 수량을 정한다. 목표는 에피소드 누적 이익을 키우는 것이다.

비교 정책은 다섯 개다.

| 정책 | 학습 | 하는 일 |
|---|---|---|
| 고정 발주 | 없음 | 매 기간 같은 수량 20 |
| 재고 수준 (base-stock) | 없음 | 재고 위치가 목표 S보다 모자라면 그 차이만큼 발주 |
| 순수 REINFORCE | 정책 경사 | 실제 할인 누적 보상으로 연속 발주 정책 학습 |
| TD Actor-Critic | 정책 경사 + 가치 함수 | TD 오차 기반 연속 발주 정책 학습 |
| DQN | 행동가치 함수 | 125개 이산 발주 조합 학습 |

## 2. 원본과 우리가 추가한 것

원본은 `vendor/gym-invmgmt` (태그 `v0.2.1`)이다. 가져온 원본 파일은 수정하지 않았다.

| 경로 | 역할 |
|---|---|
| `src/inventory_rl/env.py` | Serial-v0 생성 |
| `src/inventory_rl/policies.py` | 고정 발주, 재고 수준 기준선 |
| `src/inventory_rl/reinforce_agent.py` | 가우시안 정책과 REINFORCE 업데이트 |
| `scripts/train_reinforce.py` | 학습 |
| `scripts/compare_policies.py` | 다섯 정책을 같은 KPI로 비교 |
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
평균 = sigmoid(신경망) × 80
학습 발주 = clip(Normal(평균, 표준편차)에서 샘플링, 0, 80)
```

에피소드가 끝나면 실제 할인 누적 보상 G_t로 정책 손실 -mean(logπ × G_t)을 계산한다. 가치 함수, Advantage, entropy 보너스는 없다. 재고 수준 기준선은 시뮬레이터의 진짜 μ(t)를 보지 않고, 관측의 직전 수요만 쓴다.

## 6. 실행

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

비교 지표는 계획서와 같다. 누적 이익, 충족률, 미충족, 평균 재고, sentiment.

- 급변 수요: `outputs/trend_shock/policy_comparison_vanilla.csv`
- 정상 수요: `outputs/stationary/policy_comparison_vanilla.csv`
- 새 체크포인트: 각 수요 폴더의 `reinforce_vanilla.pt` (이전 `reinforce.pt`와 호환되지 않음)

## 7. 최신 비교 결과

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

## 8. 모델-free인지

학습 코드는 수요 식, 리드타임 식, 보상 식을 풀어 발주량을 계산하지 않는다. `reset`/`step`이 준 관측과 보상만 보고 정책을 바꾼다. 원본 환경 안에서 돌아가는 모델-free 학습이다.
