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
from inventory_rl.obs import scale_obs

# 링크 하나당 고를 수 있는 발주량. 세 링크를 조합하면 행동이 125개.
ORDER_LEVELS = (0.0, 20.0, 40.0, 60.0, 80.0)
N_LEVELS = len(ORDER_LEVELS)


def n_discrete_actions(n_links: int) -> int:
    return N_LEVELS**n_links  # 3링크면 5**3=125


def index_to_order(index: int, n_links: int = 3) -> np.ndarray:
    """0..124 를 (링크0 발주, 링크1 발주, 링크2 발주) 로 바꾼다.

    링크 순서 = 환경 행동 순서: (2,1) 소매, (3,2) 유통, (4,3) 공장.
    예: index=6 → 6=1 + 1×5 + 0×25 → (20, 20, 0)
    """
    orders = np.zeros(n_links, dtype=np.float64)
    remaining = int(index)
    for i in range(n_links):
        orders[i] = ORDER_LEVELS[remaining % N_LEVELS]  # 일의 자리부터 링크0
        remaining //= N_LEVELS  # 다음 링크는 5로 나눈 몫
    return clip_order(orders)


class ReplayBuffer:
    """최근 전이를 저장해 두고, 학습 때 무작위로 섞어 꺼낸다.

    (s, a, r, s', done) 한 줄을 전이라고 한다.
    시간 순으로만 배우면 한 에피소드에 과적합하므로 섞는다.
    """

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity  # 꽉 차면 맨 앞부터 덮어쓴다
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)  # 0..124 인덱스
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._i = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._i % self.capacity  # 예: capacity=50000, i=50000 → 칸 0 에 덮어씀
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self._i += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)  # 예: 64개 무작위 칸
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


class QNetwork(nn.Module):
    """관측 → 이산 행동 125개의 Q값.

    7강 Atari 그림의 마지막 FC 가 행동마다 Q 하나를 내던 것과 같다.
    여기 관측은 15차원 벡터라 CNN 대신 MLP 를 쓴다.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),  # 15 → 64
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),  # 64 → 125. 칸 하나가 한 이산 발주의 Q
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # 예: shape (64, 125)


class DQNAgent:
    """7강 value-based 제어. 정책은 Q 에서 입실론-그리디로 꺼낸다."""

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
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.n_links = n_links
        self.n_actions = n_discrete_actions(n_links)
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup  # 이만큼 채우기 전에는 학습 안 함
        self.target_sync = target_sync  # 200스텝마다 타깃망을 온라인망과 같게
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.reward_scale = reward_scale
        self.total_steps = 0

        self.q = QNetwork(obs_dim, self.n_actions).to(self.device)
        # 타깃 네트워크. y = r + γ max Q_target(s', a') 를 계산할 때 쓴다.
        # 온라인 Q와 같은 값이면 목표가 같이 움직여서 학습이 흔들린다.
        self.target_q = QNetwork(obs_dim, self.n_actions).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())  # 처음엔 가중치 동일
        self.target_q.eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_dim)

    def epsilon(self) -> float:
        """처음엔 거의 무작위, 갈수록 greedy. 선형으로 줄어든다."""
        # 예: 0스텝 → 1.0, 30000스텝 → 0.525, 60000스텝 이후 → 0.05
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _tensor_obs(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(scale_obs(obs), dtype=torch.float32, device=self.device)

    def select_index(self, obs: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy. greedy=True 이면 평가처럼 가장 큰 Q만 고른다."""
        if (not greedy) and (np.random.rand() < self.epsilon()):
            return int(np.random.randint(self.n_actions))  # 0..124 아무거나
        self.q.eval()
        with torch.no_grad():
            q_values = self.q(self._tensor_obs(obs).unsqueeze(0))[0]  # 길이 125
            return int(torch.argmax(q_values).item())  # Q가 가장 큰 칸의 번호

    def get_action(self, obs: np.ndarray, env, deterministic: bool = True) -> np.ndarray:
        """평가용. 이산 인덱스를 세 링크 발주량으로 바꿔 환경에 넣는다."""
        index = self.select_index(obs, greedy=deterministic)
        return index_to_order(index, n_links=self.n_links)  # 예: 6 → [20, 20, 0]

    def act_and_index(self, obs: np.ndarray) -> tuple[np.ndarray, int]:
        """학습용. 탐색이 섞인 이산 행동과 그 인덱스를 같이 준다."""
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
        self.buffer.add(scale_obs(obs), action_index, reward, scale_obs(next_obs), done)
        self.total_steps += 1

    def learn(self) -> dict[str, float]:
        """버퍼가 충분히 찬 뒤에만 Q를 한 번 맞춘다.

        목표 y = r/scale + γ max_a' Q_target(s', a')   (끝나면 γ항은 0)
        손실 = (Q(s,a) − y)²
        """
        if self.buffer.size < self.warmup:
            return {"loss": 0.0, "epsilon": self.epsilon()}  # 아직 안 배움

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rew_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device) / self.reward_scale
        done_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # gather: 125개 Q 중에서 그때 고른 행동의 Q만. 예: action=6 → Q[:, 6]
        q_sa = self.q(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next = self.target_q(next_t).max(dim=1).values  # s'에서 제일 큰 Q
            # 에피소드가 끝난 칸은 (1-1)=0 이라 미래 가치를 안 더함
            target = rew_t + self.gamma * max_next * (1.0 - done_t)
        loss = torch.nn.functional.mse_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optimizer.step()

        if self.total_steps % self.target_sync == 0:
            self.target_q.load_state_dict(self.q.state_dict())  # θ⁻ ← θ

        return {"loss": float(loss.item()), "epsilon": self.epsilon()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"q": self.q.state_dict()}, path)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.q.load_state_dict(payload["q"])
        self.target_q.load_state_dict(self.q.state_dict())  # 평가 때도 타깃을 맞춰 둠
