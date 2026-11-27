"""Sample DT training batches from recorded CartPole trajectories.

Follows the official DT repo (kzl/decision-transformer, gym/experiment.py):
trajectories are drawn with probability proportional to their length, a
window of K steps starts uniformly inside the trajectory, and short windows
are left-padded with an attention mask of 0.
"""

import numpy as np
import torch

from dt import POSITION_DIMS, RTG_SCALE


class TrajectoryDataset:
    def __init__(self, episodes, hide_velocity=False):
        self.hide_velocity = hide_velocity
        self.obs = [self._view(ep["observations"]) for ep in episodes]
        self.actions = [ep["actions"].astype(np.int64) for ep in episodes]
        self.rtg = [ep["rtg"].astype(np.float32) for ep in episodes]
        self.lengths = np.array([len(a) for a in self.actions])
        self.returns = np.array([float(ep["rewards"].sum()) for ep in episodes])

        all_obs = np.concatenate(self.obs, axis=0)
        self.state_mean = all_obs.mean(axis=0).astype(np.float32)
        self.state_std = (all_obs.std(axis=0) + 1e-6).astype(np.float32)
        self.state_dim = all_obs.shape[1]
        self.p_sample = self.lengths / self.lengths.sum()

    def _view(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return obs[:, POSITION_DIMS] if self.hide_velocity else obs

    def sample_batch(self, batch_size, K, rng, n_actions=2):
        """Return (states, actions, rtg, timesteps, mask) tensors.

        Shapes: states B×K×S, actions B×K (long), rtg B×K×1,
        timesteps B×K (long), mask B×K (1 = real step, 0 = left padding).
        """
        idx = rng.choice(len(self.lengths), size=batch_size, p=self.p_sample)
        S = self.state_dim
        states = np.zeros((batch_size, K, S), dtype=np.float32)
        actions = np.full((batch_size, K), n_actions, dtype=np.int64)
        rtg = np.zeros((batch_size, K, 1), dtype=np.float32)
        timesteps = np.zeros((batch_size, K), dtype=np.int64)
        mask = np.zeros((batch_size, K), dtype=np.float32)

        for b, i in enumerate(idx):
            T = self.lengths[i]
            si = rng.integers(0, T)
            ei = min(si + K, T)
            n = ei - si
            states[b, K - n :] = (self.obs[i][si:ei] - self.state_mean) / self.state_std
            actions[b, K - n :] = self.actions[i][si:ei]
            rtg[b, K - n :, 0] = self.rtg[i][si:ei] / RTG_SCALE
            timesteps[b, K - n :] = np.arange(si, ei)
            mask[b, K - n :] = 1.0

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(rtg),
            torch.from_numpy(timesteps),
            torch.from_numpy(mask),
        )
