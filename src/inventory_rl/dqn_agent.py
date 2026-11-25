"""DQN으로 이산 발주를 학습한다. (수업 7강 슬라이드 14)

환경 발주량은 연속이라 표 Q-learning을 그대로 쓸 수 없다.
링크마다 발주를 {0, 20, 40, 60, 80} 다섯 칸으로 자르면
행동이 5×5×5 = 125개가 된다. DQN은 이 125칸의 Q값을 신경망으로 근사한다.

    Q(s, a) ≈ “관측 s에서 이산 발주 a를 고르면 앞으로 받을 보상”

7강 DQN 절차와 코드 대응:
    1) ε-greedy 로 a 선택          → select_index
    2) (s, a, r, s') 를 버퍼에 저장 → ReplayBuffer
    3) 고정 타깃 θ⁻ 로 y 계산      → target_q
    4) MSE (y − Q(s,a;θ))²         → learn()

    L(θ) = E[ ( r + γ max_{a'} Q(s', a'; θ⁻) − Q(s, a; θ) )² ]

원본 환경은 연속 발주다. DQN을 쓰려고 이산 칸으로 바꾼 것이지,
연속 행동 그대로 DQN을 돌린 것은 아니다. 보상/50 은 슬라이드에 없고 스케일용이다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from inventory_rl.env import clip_order
from inventory_rl.obs import scale_obs, tensor_obs
from inventory_rl.model_utils import select_device, make_backbone, update_parameters, save_weights, load_weights

# 링크 하나당 고를 수 있는 발주량. 세 링크를 조합하면 행동이 125개.
ORDER_LEVELS = (0.0, 20.0, 40.0, 60.0, 80.0)
N_LEVELS = len(ORDER_LEVELS)


def n_discrete_actions(n_links: int) -> int:
    # 링크마다 5칸. Serial 3링크 → 5**3=125. DQN 출력 칸 수.
    return N_LEVELS**n_links


def index_to_order(index: int, n_links: int = 3) -> np.ndarray:
    """신경망이 고른 번호 0~124 를 환경에 넣을 발주 3칸으로 바꾼다.

    DQN 출력은 [20, 20, 0] 같은 수량이 아니다. 125개 조합 중 몇 번인지다.
    링크마다 고를 수 있는 양은 다섯 칸: 칸 번호 0,1,2,3,4 → 발주 0,20,40,60,80.
    세 링크를 한꺼번에 고르므로 5×5×5=125. 그 번호가 index.

    번호를 5진수처럼 푼다. 일의 자리=첫 링크, 5의 자리=둘째, 25의 자리=셋째.
      actions[0]  (2,1) 유통→소매
      actions[1]  (3,2) 공장→유통
      actions[2]  (4,3) 원자재→공장

    예:
      index=0   → 칸 (0,0,0) → 발주 (0,  0,  0)
      index=1   → 칸 (1,0,0) → 발주 (20, 0,  0)   소매만 20
      index=5   → 칸 (0,1,0) → 발주 (0,  20, 0)   유통만 20
      index=6   → 칸 (1,1,0) → 발주 (20, 20, 0)
                  6 = 1 + 1×5 + 0×25. 칸 1→20, 칸 1→20, 칸 0→0
      index=124 → 칸 (4,4,4) → 발주 (80, 80, 80)
    """
    orders = np.zeros(n_links, dtype=np.float64)
    remaining = int(index)  # 예: 6. 아래에서 5로 나누며 자릿수를 떼어 낸다.
    for i in range(n_links):
        # 지금 자릿수 = remaining 을 5로 나눈 나머지. 그게 이 링크의 칸 번호.
        # 예 index=6, i=0: 6%5=1 → ORDER_LEVELS[1]=20  (유통→소매)
        #              i=1: 1%5=1 → 20                   (공장→유통)
        #              i=2: 0%5=0 → 0                    (원자재→공장)
        orders[i] = ORDER_LEVELS[remaining % N_LEVELS]
        remaining //= N_LEVELS  # 다음 자릿수. 6//5=1, 1//5=0
    return clip_order(orders)  # 0~80. 이미 그 안이라 값은 그대로.


class ReplayBuffer:
    """최근 전이를 저장해 두고, 학습 때 무작위로 섞어 꺼낸다.

    (s, a, r, s', done) 한 줄을 전이라고 한다.
    시간 순으로만 배우면 한 에피소드에 과적합하므로 섞는다.
    """

    def __init__(self, capacity: int, obs_dim: int) -> None:
        # 최대 몇 줄을 담을지. 기본 50000. 넘치면 add() 가 칸 0부터 덮어쓴다.
        self.capacity = capacity
        # 아래 다섯 배열은 같은 행 번호가 한 전이 (s, a, r, s', done).
        # 예: 학습 기본 obs_dim=15, capacity=50000
        #     obs[3] = 기간 t 관측, next_obs[3] = 그 다음 관측, actions[3] = 그때 고른 번호
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)  # s. 행=전이, 열=관측 15칸
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)  # s'
        self.actions = np.zeros(capacity, dtype=np.int64)  # a. 발주 벡터가 아니라 0~124 번호
        self.rewards = np.zeros(capacity, dtype=np.float32)  # r. 그 기간 보상
        self.dones = np.zeros(capacity, dtype=np.float32)  # 에피소드 끝이면 1. 아니면 0
        self._i = 0  # 다음에 쓸 칸. 0,1,2,... 계속 증가. 실제 칸은 _i % capacity
        self.size = 0  # 지금 들어 있는 줄 수. capacity 를 넘지 않음. sample 범위가 0..size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        # 원형 버퍼. _i 가 50000 이면 칸 0 을 덮어쓰고, 50001 이면 칸 1.
        i = self._i % self.capacity
        self.obs[i] = obs  # 정규화된 s. remember() 가 scale_obs 한 값
        self.next_obs[i] = next_obs  # s'
        self.actions[i] = action  # 예: 6. 발주량 [20,20,0] 이 아님
        self.rewards[i] = reward  # 환경이 준 한 기간 이익. 아직 /50 안 함
        self.dones[i] = float(done)  # 기간 29 step 이면 True → 1.0
        self._i += 1
        # 줄 수는 capacity 에서 멈춤. 덮어써도 size 는 50000 유지.
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        # 들어 있는 줄 0..size 에서 64개를 복원추출. 시간 순이 아님.
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


class QNetwork(nn.Module):
    """행동가치 Q(s,a) 를 신경망으로 근사한다.

    입력은 관측 s 하나. 출력은 125칸. k번 칸 = Q(s, index=k).
    a 를 입력에 붙이지 않는다. 한 번에 모든 이산 발주의 Q 를 낸다.
    7강 Atari 마지막 FC 가 행동마다 Q 하나를 내던 것과 같다.
    여기 관측은 15차원 벡터라 CNN 대신 MLP 를 쓴다.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        # 공통 은닉층 뒤에 Q값 출력층을 붙인다 (64 → 125).
        self.net = make_backbone(obs_dim, hidden)
        self.net.append(nn.Linear(hidden, n_actions))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 배치면 (64, 15) → (64, 125). 한 줄이면 unsqueeze 후 (1, 125).
        return self.net(obs)


class DQNAgent:
    """7강 value-based 제어. 정책 네트워크는 없고, Q 에서 ε-greedy 로 행동을 꺼낸다."""

    def __init__(
        self,
        obs_dim: int = 15,
        n_links: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        warmup: int = 1000,
        target_sync: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 60_000,
        reward_scale: float = 50.0,
        device: str | None = None,
    ) -> None:
        self.device = select_device(device)
        self.n_links = n_links  # Serial 발주 링크 3
        self.n_actions = n_discrete_actions(n_links)  # 125
        self.gamma = gamma  # 할인. y = r + γ max Q(s',·)
        self.batch_size = batch_size  # learn() 한 번에 꺼내는 전이 수
        self.warmup = warmup  # 이만큼 채우기 전에는 학습 안 함
        self.target_sync = target_sync  # 200스텝마다 타깃망을 온라인망과 같게
        self.epsilon_start = epsilon_start  # 처음엔 전부 무작위
        self.epsilon_end = epsilon_end  # 나중엔 20번에 1번만 무작위
        self.epsilon_decay_steps = epsilon_decay_steps  # 그 사이 6만 스텝
        self.reward_scale = reward_scale  # 보상 대략 ±수십이라 50으로 나눔
        self.total_steps = 0  # remember() 호출 횟수. ε 감소·타깃 동기화에 씀

        self.q = QNetwork(obs_dim, self.n_actions).to(self.device)  # 온라인망. 학습되는 θ
        # 타깃 네트워크. y = r + γ max Q_target(s', a') 를 계산할 때 쓴다.
        # 온라인 Q와 같은 값이면 목표가 같이 움직여서 학습이 흔들린다.
        self.target_q = QNetwork(obs_dim, self.n_actions).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())  # 처음엔 가중치 동일
        self.target_q.eval()  # 타깃은 경사 안 탐. 200스텝마다 복사만
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)  # θ 만 갱신
        self.buffer = ReplayBuffer(buffer_size, obs_dim)

    def epsilon(self) -> float:
        """지금 ε. select_index 가 이 확률로 0~124 를 무작위로 고른다.

        기본값: 시작 1.0, 끝 0.05, 6만 스텝에 걸쳐 직선으로 감소.
        1.0 이면 항상 무작위(탐험). 0 이면 항상 Q가 큰 칸(활용).
        학습이 진행될수록 탐험을 줄인다. 6만 스텝 이후에는 0.05 로 고정.
        """
        # 감소를 얼마나 했는지 0~1. 6만을 넘으면 1 로 자른다.
        # 예: 0스텝 → 0,  30000스텝 → 0.5,  60000스텝 → 1,  90000스텝 → 1
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        # 시작 + 진행비율×(끝−시작). 끝<시작이라 frac 이 커질수록 ε 가 작아진다.
        # 예: 0스텝   → 1.0 + 0  ×(0.05−1.0) = 1.0    전부 무작위
        #     30000   → 1.0 + 0.5×(−0.95)     = 0.525  절반 정도 무작위
        #     60000 이후 → 1.0 + 1.0×(−0.95)   = 0.05   20번에 1번만 무작위
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_index(self, obs: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy. 반환은 발주량이 아니라 0~124 번호.

        greedy=False (학습): 확률 ε 로 아무 번호, 아니면 Q가 큰 번호.
        greedy=True  (평가): ε 무시하고 항상 argmax Q.
        """
        if (not greedy) and (np.random.rand() < self.epsilon()):
            return int(np.random.randint(self.n_actions))  # 0..124 아무거나
        self.q.eval()
        with torch.no_grad():  # 행동 고를 때는 경사 안 쌓음
            # unsqueeze(0): (15,) → (1,15). [0] 로 배치 차원 제거 → 길이 125
            q_values = self.q(tensor_obs(obs, self.device).unsqueeze(0))[0]
            return int(torch.argmax(q_values).item())  # Q가 가장 큰 칸의 번호. 예: 6

    def get_action(self, obs: np.ndarray, env, deterministic: bool = True) -> np.ndarray:
        """평가·비교용. compare_policies 가 이 함수를 부른다.

        deterministic=True 이면 탐험 없이 argmax Q.
        env 인자는 인터페이스 맞추기용. 여기선 안 쓴다.
        """
        index = self.select_index(obs, greedy=deterministic)
        return index_to_order(index, n_links=self.n_links)  # 예: 6 → [20, 20, 0]

    def act_and_index(self, obs: np.ndarray) -> tuple[np.ndarray, int]:
        """학습 루프용. train_dqn 이 step 전에 이걸 부른다.

        발주 벡터는 env.step 에, 번호는 remember 에 넣어야 해서 둘 다 준다.
        greedy=False 라서 ε 탐험이 섞인다.
        """
        index = self.select_index(obs, greedy=False)
        return index_to_order(index, n_links=self.n_links), index

    def remember(
        self,
        obs: np.ndarray,
        action_index: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        # 버퍼에는 정규화된 관측을 넣는다. 학습 때도 같은 스케일로 보게.
        self.buffer.add(scale_obs(obs), action_index, reward, scale_obs(next_obs), done)
        self.total_steps += 1  # ε 감소, 타깃 200스텝 복사용

    def learn(self) -> dict[str, float]:
        """버퍼가 충분히 찬 뒤에만 Q를 한 번 맞춘다.

        목표 y = r/scale + γ max_a' Q_target(s', a')   (끝나면 γ항은 0)
        손실 = (Q(s,a) − y)²
        한 번 호출 = 배치 64개로 θ 를 한 스텝 갱신. 에피소드 전체가 아님.
        """
        # warmup=1000. 전이가 1000줄 찰 때까지는 Q 를 안 맞춘다.
        # 줄이 적으면 sample(64) 이 같은 전이를 반복해서 보고,
        # 초반은 ε≈1 이라 무작위 발주라서 Q 가 쓰레기 경험에 맞춰진다.
        # 에피소드 30기간이면 약 33판(1000/30) 동안은 탐험·저장만.
        # 이 return 이면 아래 sample·MSE·optimizer.step 은 실행 안 됨.
        if self.buffer.size < self.warmup:
            # loss 0.0 은 “아직 안 배움” 표시. 오차가 진짜 0 인 게 아님.
            # epsilon 은 로그용. remember() 마다 ε 는 줄어든다.
            return {"loss": 0.0, "epsilon": self.epsilon()}

        # select_index 가 eval() 로 두었을 수 있다. 학습 전에 train 모드로 되돌린다.
        # (Dropout/BN 은 없지만 딥러닝 학습 템플릿에 맞춤. target_q 는 계속 eval.)
        self.q.train()

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)  # (64, 15)
        next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)  # (64,) 예: 6
        # 보상 대략 수십. 50으로 나눠 Q 스케일을 1 근처로.
        rew_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device) / self.reward_scale
        done_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)  # 0 또는 1

        # 손실에 필요한 건 “그때 고른 a 의 Q” 하나다. 125칸 전부가 아님.
        # self.q(obs_t) 모양 (64, 125):
        #   행 = 배치의 관측, 열 = 이산 발주 번호 0~124
        #   예: 2행·3칸짜리로 줄이면
        #        Q = [[1.1, 0.2, 3.0],     ← 관측0, 번호0/1/2 의 Q
        #             [0.5, 2.2, 0.1]]     ← 관측1
        #        그때 고른 번호 act_t = [2, 0]
        #        필요한 값 = [3.0, 0.5]  (0행의 2열, 1행의 0열)
        # gather(1, 인덱스): 각 행에서 “인덱스에 적힌 열”만 뽑는다.
        #   dim=1 이 “열 방향”. 인덱스는 Q 와 같은 2차원이어야 해서
        #   unsqueeze(1): [2, 0] → [[2], [0]]
        #   gather 결과: [[3.0], [0.5]]
        #   squeeze(1): [3.0, 0.5]  ← 아래 target (64,) 과 맞춰 MSE
        # 실제: act_t 에 6 이 있으면 그 행의 6번 칸 = 발주 (20,20,0) 의 Q.
        q_sa = self.q(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():  # y 는 상수처럼. 타깃망 경사는 안 흐름
            # target_q(next_t): (64, 125) → 행마다 최댓값. “s'에서 제일 좋은 발주”
            max_next = self.target_q(next_t).max(dim=1).values
            # done=1 이면 (1-1)=0 이라 미래 Q 를 안 더함. 에피소드가 거기서 끝이라서.
            # 예: r/50=0.6, γ=0.99, max_next=2.0, done=0 → y=0.6+0.99×2=2.58
            target = rew_t + self.gamma * max_next * (1.0 - done_t)
        loss = torch.nn.functional.mse_loss(q_sa, target)  # 평균 (Q(s,a) − y)²

        update_parameters(self.q, self.optimizer, loss)

        if self.total_steps % self.target_sync == 0:
            self.target_q.load_state_dict(self.q.state_dict())  # θ⁻ ← θ. 200스텝마다

        return {"loss": float(loss.item()), "epsilon": self.epsilon()}

    def save(self, path: str | Path) -> None:
        save_weights(self.q, path, key="q")

    def load(self, path: str | Path) -> None:
        load_weights(self.q, path, key="q", device=self.device)
        self.target_q.load_state_dict(self.q.state_dict())  # 평가 때도 타깃을 맞춰 둠
