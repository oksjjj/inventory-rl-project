"""이산 정책의 실제 Serial-v0 학습, 행동 매핑, 저장/로드 검사."""
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inventory_rl.discrete_agent import DiscreteReinforceAgent, DiscreteActorCriticAgent
from inventory_rl.dqn_agent import ORDER_LEVELS, n_discrete_actions, index_to_order
from inventory_rl.env import make_serial_env, clip_order


def check_order(action):
    assert action.shape == (3,)
    assert np.isin(action, ORDER_LEVELS).all()


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    assert n_discrete_actions(3) == 125
    orders = [tuple(index_to_order(i)) for i in range(125)]
    assert len(set(orders)) == 125
    assert orders[6] == (20, 20, 0)
    assert orders[-1] == (80, 80, 80)

    with tempfile.TemporaryDirectory() as tmp:
        for demand in ("stationary", "trend_shock"):
            for agent_cls in (DiscreteReinforceAgent, DiscreteActorCriticAgent):
                env = make_serial_env(demand=demand)
                try:
                    obs, _ = env.reset(seed=42)
                    agent = agent_cls(device="cpu")
                    with torch.no_grad():
                        output = agent.model(agent._tensor_obs(obs).unsqueeze(0))
                        logits = output[0] if isinstance(output, tuple) else output
                        assert logits.shape == (1, 125)
                        probs = Categorical(logits=logits).probs
                        assert torch.allclose(probs, torch.full_like(probs, 1 / 125))

                    initial = agent.model.policy_head.weight.detach().clone()
                    log_probs, rewards = [], []
                    done = False
                    agent.model.train()
                    while not done:
                        result = agent.act_and_log(obs)
                        action, log_prob = result[:2]
                        check_order(action)
                        next_obs, reward, terminated, truncated, _ = env.step(clip_order(action))
                        done = terminated or truncated
                        rewards.append(float(reward))
                        if isinstance(agent, DiscreteReinforceAgent):
                            log_probs.append(log_prob)
                        else:
                            _, _, entropy, value = result
                            stats = agent.learn_step(log_prob, entropy, value, float(reward), next_obs, done)
                            assert all(np.isfinite(v) for v in stats.values())
                        obs = next_obs
                    if isinstance(agent, DiscreteReinforceAgent):
                        stats = agent.update_episode(log_probs, rewards)
                        assert all(np.isfinite(v) for v in stats.values())
                    assert len(rewards) == 30
                    assert not torch.equal(initial, agent.model.policy_head.weight.detach())

                    obs, _ = env.reset(seed=43)
                    before = agent.get_action(obs, env)
                    check_order(before)
                    path = Path(tmp) / f"{demand}_{agent_cls.__name__}.pt"
                    agent.save(path)
                    loaded = agent_cls(device="cpu")
                    loaded.load(path)
                    assert np.array_equal(before, loaded.get_action(obs, env))
                    print(f"OK: {demand}, {agent_cls.__name__}, steps={len(rewards)}")
                finally:
                    env.close()
    print("Discrete policy smoke test passed.")


if __name__ == "__main__":
    main()
