"""Record every completed episode of an agent's interaction with a Gymnasium env.

The wrapper stores, for each episode, the observation *before* each action
(s_t) paired with the action (a_t) and the reward (r_t) it produced.  Episode
ends are detected from ``terminated or truncated``, and only completed
episodes are appended, so a final, unfinished episode is discarded.
"""

import gymnasium as gym
import numpy as np


class EpisodeRecorder(gym.Wrapper):
    """Collect episodes as trajectory dicts while stepping the wrapped env."""

    def __init__(self, env):
        super().__init__(env)
        self.episodes = []
        self._current = None
        self._obs = None

    def reset(self, **kwargs):
        # Drop any partial episode from a previous, unfinished rollout.
        self._current = None
        obs, info = self.env.reset(**kwargs)
        self._obs = np.asarray(obs, dtype=np.float32)
        self._current = {"observations": [], "actions": [], "rewards": []}
        return obs, info

    def step(self, action):
        # Pair the pre-step observation s_t with (a_t, r_t).
        if self._current is not None:
            self._current["observations"].append(self._obs)
            self._current["actions"].append(int(np.asarray(action).item()))

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._current is not None:
            self._current["rewards"].append(float(reward))

        self._obs = np.asarray(obs, dtype=np.float32)

        if terminated or truncated:
            self._finish_episode(terminated, truncated)

        return obs, reward, terminated, truncated, info

    def _finish_episode(self, terminated, truncated):
        cur, self._current = self._current, None
        if cur is None or not cur["actions"]:
            return

        actions = np.asarray(cur["actions"], dtype=np.int64)
        T = len(actions)

        dones = np.zeros(T, dtype=bool)
        dones[-1] = True
        terminals = np.zeros(T, dtype=bool)
        terminals[-1] = bool(terminated)
        timeouts = np.zeros(T, dtype=bool)
        timeouts[-1] = bool(truncated)

        self.episodes.append(
            {
                "observations": np.stack(cur["observations"]).astype(
                    np.float32
                ),  # T x 4, full state
                "actions": actions,  # T
                "rewards": np.asarray(cur["rewards"], dtype=np.float32),  # T
                "dones": dones,  # True at the last step
                "terminals": terminals,  # True at the last step if terminated
                "timeouts": timeouts,  # True at the last step if truncated
                # Return-to-go: CartPole rewards are 1 per step, so T - t.
                "rtg": np.arange(T, 0, -1, dtype=np.float32),
            }
        )
