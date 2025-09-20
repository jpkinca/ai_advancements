from types import SimpleNamespace
import pytest
from tensortrade_risk_module import RiskAwareReward, RiskAwareRewardConfig


class DummyEnv(SimpleNamespace):
    pass


def test_risk_aware_reward_progression():
    reward = RiskAwareReward(RiskAwareRewardConfig(window=5))
    env = DummyEnv(net_worth=10000)
    reward.env = env
    reward.reset()
    r0 = reward.get_reward(0)
    env.net_worth = 10050
    r1 = reward.get_reward(0.1)
    env.net_worth = 10025
    r2 = reward.get_reward(0.2)
    assert r0 == 0.0
    for r in (r1, r2):
        assert isinstance(r, float)
