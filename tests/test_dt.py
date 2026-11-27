import gymnasium as gym
import numpy as np
import pytest
import torch

from dt.model import DecisionTransformer
from recorder import EpisodeRecorder


@torch.no_grad()
@pytest.mark.parametrize("use_rtg", [True, False])
def test_no_action_leakage(use_rtg):
    model = DecisionTransformer(4, K=5, hidden=16, n_layer=1, dropout=0.0, use_rtg=use_rtg).eval()
    s, a, r = torch.randn(1, 5, 4), torch.zeros(1, 5, dtype=torch.long), torch.randn(1, 5, 1)
    t, mask = torch.arange(5)[None], torch.ones(1, 5)
    logits = model(s, a, r, t, mask)
    a[:, 2] = 1
    assert torch.allclose(logits[:, 2], model(s, a, r, t, mask)[:, 2], atol=1e-5)


@torch.no_grad()
def test_padding_invariance():
    model = DecisionTransformer(state_dim=4, K=5, hidden=16, n_layer=1, dropout=0.0).eval()
    s, a, r = torch.randn(1, 5, 4), torch.tensor([[2, 2, 0, 1, 0]]), torch.randn(1, 5, 1)
    t, mask = torch.tensor([[0, 0, 0, 1, 2]]), torch.tensor([[0, 0, 1, 1, 1]])
    logits = model(s, a, r, t, mask)
    s[:, :2], a[:, :2], r[:, :2] = -s[:, :2], 1, -r[:, :2]
    assert torch.allclose(logits[:, 2:], model(s, a, r, t, mask)[:, 2:], atol=1e-5)


def test_recorder():
    class Env(gym.Env):
        observation_space = gym.spaces.Box(0, 3, (4,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)

        def reset(self):
            self.count = 0
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            self.count += 1
            return np.full(4, self.count, dtype=np.float32), 1.0, self.count == 3, False, {}

    recorder = EpisodeRecorder(Env())
    recorder.reset()
    for action in [0, 1, 0]:
        recorder.step(action)
    (ep,) = recorder.episodes
    assert ep["actions"].tolist() == [0, 1, 0]
    assert ep["rtg"].tolist() == [3, 2, 1]
    assert ep["observations"][:, 0].tolist() == [0, 1, 2]
    assert ep["dones"].tolist() == [False, False, True]
    assert ep["terminals"][-1]
