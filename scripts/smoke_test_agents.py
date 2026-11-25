"""3개 에이전트의 공통 함수 연결 및 학습·저장/복원 검사.

실제 Serial-v0 환경에서 CPU로 한 에피소드씩 실행한다.
정식 실험 결과 파일은 생성하지 않는다.
"""
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_rl.dqn_agent import DQNAgent
from inventory_rl.reinforce_agent import ReinforceAgent
from inventory_rl.actor_critic_agent import ActorCriticAgent
from inventory_rl.env import make_serial_env, clip_order
from inventory_rl.obs import scale_obs, tensor_obs


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        for demand in ("stationary", "trend_shock"):
            for agent_cls in (DQNAgent, ReinforceAgent, ActorCriticAgent):
                env = make_serial_env(demand=demand)
                try:
                    obs, _ = env.reset(seed=42)
                    # DQN은 짧은 검사에서도 실제 경사 갱신을 하도록 warmup만 낮춘다.
                    kwargs = {"device": "cpu"}
                    if agent_cls is DQNAgent:
                        kwargs.update(warmup=2, batch_size=2)
                    agent = agent_cls(**kwargs)
                    model = agent.q if agent_cls is DQNAgent else agent.model
                    initial = {k: v.clone() for k, v in model.state_dict().items()}
                    assert torch.equal(tensor_obs(obs, agent.device),
                                       torch.as_tensor(scale_obs(obs), dtype=torch.float32))
                    # 기존 체크포인트와 동일한 state_dict 이름을 유지하는지 확인한다.
                    if agent_cls is DQNAgent:
                        expected = {f"net.{i}.{p}" for i in (0, 2, 4) for p in ("weight", "bias")}
                    else:
                        expected = {"log_std", "mean_head.weight", "mean_head.bias"}
                        expected |= {f"backbone.{i}.{p}" for i in (0, 2) for p in ("weight", "bias")}
                        if agent_cls is ActorCriticAgent:
                            expected |= {"value_head.weight", "value_head.bias"}
                    assert set(initial) == expected
                    log_probs, rewards = [], []
                    done = False
                    while not done:
                        if agent_cls is DQNAgent:
                            action, index = agent.act_and_index(obs)
                        else:
                            result = agent.act_and_log(obs)
                            action, log_prob = result[:2]
                        next_obs, reward, terminated, truncated, _ = env.step(clip_order(action))
                        done = terminated or truncated
                        rewards.append(float(reward))
                        if agent_cls is DQNAgent:
                            agent.remember(obs, index, float(reward), next_obs, done)
                            stats = agent.learn()
                        elif agent_cls is ActorCriticAgent:
                            _, _, entropy, value = result
                            stats = agent.learn_step(log_prob, entropy, value, float(reward), next_obs, done)
                        else:
                            log_probs.append(log_prob)
                        obs = next_obs
                    if agent_cls is ReinforceAgent:
                        stats = agent.update_episode(log_probs, rewards)
                        # REINFORCE는 정책 손실과 원본 누적 보상만 반환한다.
                        assert set(stats) == {"loss", "return"}
                        assert stats["return"] == float(sum(rewards))
                    assert len(rewards) == 30
                    assert all(np.isfinite(v) for v in stats.values())
                    assert any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
                    obs, _ = env.reset(seed=43)
                    expected_action = agent.get_action(obs, env)
                    path = Path(tmp) / f"{demand}_{agent_cls.__name__}.pt"
                    agent.save(path)
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                    assert set(checkpoint) == ({"q"} if agent_cls is DQNAgent else {"model"})
                    loaded = agent_cls(**kwargs)
                    loaded.load(path)
                    assert np.array_equal(expected_action, loaded.get_action(obs, env))
                    print(f"OK: {demand}, {agent_cls.__name__}, steps={len(rewards)}")
                finally:
                    env.close()
    print("Agent smoke test passed.")


if __name__ == "__main__":
    main()
