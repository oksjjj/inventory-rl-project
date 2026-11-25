"""DQN과 동일한 125개 이산 발주 조합을 사용하는 정책 경사 에이전트.

REINFORCE: 실제 할인 누적 보상으로 에피소드 끝에 정책만 갱신.
Actor-Critic: 매 기간 TD 오차로 정책과 가치 함수를 갱신.
두 정책 모두 Categorical 분포를 사용한다.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from inventory_rl.dqn_agent import index_to_order, n_discrete_actions
from inventory_rl.reinforce_agent import ReinforceAgent
from inventory_rl.actor_critic_agent import ActorCriticAgent


class DiscretePolicyNet(nn.Module):
    """상태별 125개 행동의 logit(정규화 전 점수)을 출력한다."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        # (B,15) → (B,64) → (B,64) → (B,125).
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        # W,b를 모두 0으로 초기화하면 모든 상태에서 초기 확률이 1/125.
        # DQN의 초기 epsilon=1과 동일한 균등 탐험 조건이다.
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Categorical(logits=...) 내부에서 softmax에 해당하는 정규화를 한다.
        return self.policy_head(self.backbone(obs))


class DiscreteReinforceAgent(ReinforceAgent):
    """부모의 Monte Carlo 학습, 관측 전처리, 저장/로드를 재사용한다."""

    def __init__(self, obs_dim=15, act_dim=3, lr=3e-4, gamma=0.99,
                 reward_scale=50.0, device=None):
        # 부모의 Gaussian 모델을 만들지 않고 이산 정책망만 생성한다.
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.n_links = act_dim
        self.n_actions = n_discrete_actions(act_dim)
        self.model = DiscretePolicyNet(obs_dim, self.n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def act_and_log(self, obs: np.ndarray):
        # 1. 정규화된 상태에서 125개 logits를 계산한다.
        logits = self.model(self._tensor_obs(obs).unsqueeze(0))
        # 2. 각 조합의 선택 확률 exp(z_i)/Σ exp(z_j)를 구성한다.
        dist = Categorical(logits=logits)
        # 3. 세 링크를 하나의 조합 번호로 함께 샘플링한다.
        index = dist.sample()
        # 4. 이산 행동의 로그확률(밀도가 아님). 역전파 그래프를 유지한다.
        log_prob = dist.log_prob(index)
        # 5. DQN과 동일한 변환. 예: 6 → [20,20,0].
        action = index_to_order(int(index.item()), n_links=self.n_links)
        return action, log_prob

    def get_action(self, obs, env, deterministic=True):
        # 평가에서는 확률이 최대인 이산 행동을 선택한다.
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._tensor_obs(obs).unsqueeze(0))
            index = logits.argmax(dim=-1) if deterministic else Categorical(logits=logits).sample()
        return index_to_order(int(index.item()), n_links=self.n_links)


class DiscreteActorCriticNet(DiscretePolicyNet):
    """동일한 Actor 정책망에 상태 가치 V(s) head를 추가한다."""

    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__(obs_dim, n_actions, hidden)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        hidden = self.backbone(obs)
        logits = self.policy_head(hidden)  # (B,125)
        value = self.value_head(hidden).squeeze(-1)  # (B,)
        return logits, value


class DiscreteActorCriticAgent(ActorCriticAgent):
    """연속형과 같은 TD 학습 신호를 사용하며 Actor만 Categorical로 바꾼다."""

    def __init__(self, obs_dim=15, act_dim=3, lr=3e-4, gamma=0.99,
                 entropy_coef=0.01, reward_scale=50.0, device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.reward_scale = reward_scale
        self.n_links = act_dim
        self.n_actions = n_discrete_actions(act_dim)
        self.model = DiscreteActorCriticNet(obs_dim, self.n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def act_and_log(self, obs):
        logits, value = self.model(self._tensor_obs(obs).unsqueeze(0))
        dist = Categorical(logits=logits)
        index = dist.sample()
        log_prob = dist.log_prob(index)
        # H=-Σπ logπ. 초기 균등 분포에서는 log(125).
        entropy = dist.entropy()
        action = index_to_order(int(index.item()), n_links=self.n_links)
        return action, log_prob, entropy, value.squeeze(0)

    def get_action(self, obs, env, deterministic=True):
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(self._tensor_obs(obs).unsqueeze(0))
            index = logits.argmax(dim=-1) if deterministic else Categorical(logits=logits).sample()
        return index_to_order(int(index.item()), n_links=self.n_links)

    def learn_step(self, log_prob, entropy, value, reward, next_obs, done):
        # TD 목표 y=r/50+γV(s'). 종료 시 미래 가치는 0.
        # 목표값을 상수로 취급하여 다음 상태 쪽으로 역전파하지 않는다.
        with torch.no_grad():
            if done:
                bootstrap = 0.0
            else:
                _, next_value = self.model(self._tensor_obs(next_obs).unsqueeze(0))
                bootstrap = float(next_value.item())
        td_target = reward / self.reward_scale + self.gamma * bootstrap
        td_error = td_target - value

        # Actor의 정책 경사는 δ를 상수로 취급한다.
        # Critic은 V(s)가 TD 목표를 맞추도록 학습한다.
        # 기존 연속형과 동일한 entropy 계수, 가치 손실 가중치를 사용한다.
        policy_loss = -(log_prob.squeeze() * td_error.detach())
        value_loss = td_error.pow(2)
        entropy_loss = -entropy.squeeze()
        loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "td_error": float(td_error.detach().item()),
        }
