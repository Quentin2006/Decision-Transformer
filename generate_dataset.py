"""Train a DQN on CartPole-v1 and record its training episodes as a dataset.

The DQN is only a data generator: every completed training episode is kept,
tagged with return-to-go, in the format used by the Decision Transformer.

By default a live pygame window plays every training episode as it happens
(``render_mode="human"``); ``--no-live`` trains headless.  Rendering never
touches the RNG or the physics, so the recorded dataset is identical either
way; only wall-clock time differs.
"""

import argparse
import os
import pickle

import gymnasium as gym
import matplotlib
import pygame

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

from recorder import EpisodeRecorder

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PICKLE_PATH = os.path.join(DATA_DIR, "cartpole_dqn.pkl")
PLOT_PATH = os.path.join(DATA_DIR, "returns.png")

# CartPole's native metadata["render_fps"]; --speed 1 plays at this rate.
CARTPOLE_NATIVE_FPS = 50


class LiveEpisodeRecorder(EpisodeRecorder):
    """EpisodeRecorder that also reports each completed episode."""

    def __init__(self, env, on_episode_end):
        super().__init__(env)
        self._on_episode_end = on_episode_end

    def _finish_episode(self, terminated, truncated):
        before = len(self.episodes)
        super()._finish_episode(terminated, truncated)
        if len(self.episodes) > before and self._on_episode_end:
            self._on_episode_end(len(self.episodes), self.episodes[-1])


def live_caption(episodes_done, episode):
    """Set the title: DQN training — episode N — last return R.

    After episode k finishes, the window is playing episode k+1, whose
    return is not yet known, so the title shows k+1 with R_k.
    """
    ret = float(episode["rewards"].sum())
    pygame.display.set_caption(
        f"DQN training — episode {episodes_done + 1} — last return {ret:.0f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="total environment steps for DQN training",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="train headless, without the live window",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=4.0,
        help="live-window playback multiplier, effective fps = 50*SPEED "
        "(1 = real time; ignored with --no-live)",
    )
    args = parser.parse_args()
    if not args.speed > 0:
        parser.error(f"--speed must be > 0, got {args.speed}")
    return args


def main():
    args = parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)

    # The live window only renders: it never touches the RNG or the physics,
    # so the DQN experiences — and the recorder records — exactly the same
    # data with or without it.
    if args.no_live:
        recorder = EpisodeRecorder(gym.make("CartPole-v1"))
    else:
        env = gym.make("CartPole-v1", render_mode="human")
        # Set the playback rate on the unwrapped env before the first render:
        # CartPole ticks its clock at metadata["render_fps"] per frame.
        env.unwrapped.metadata = {
            **env.unwrapped.metadata,
            "render_fps": CARTPOLE_NATIVE_FPS * args.speed,
        }
        recorder = LiveEpisodeRecorder(env, live_caption)

    # Seed the environment once; subsequent unseeded resets continue this
    # RNG stream, so the whole run is reproducible.
    recorder.reset(seed=args.seed)

    if not args.no_live:
        # reset() has rendered, so the window exists; episode 1 is on screen.
        pygame.display.set_caption("DQN training — episode 1")

    # RL Zoo's CartPole-v1 DQN hyperparameters.
    model = DQN(
        "MlpPolicy",
        recorder,
        seed=args.seed,
        learning_rate=2.3e-3,
        batch_size=64,
        buffer_size=100_000,
        learning_starts=1000,
        gamma=0.99,
        target_update_interval=10,
        train_freq=256,
        gradient_steps=128,
        exploration_fraction=0.16,
        exploration_final_eps=0.04,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        device="cpu",
    )
    try:
        model.learn(total_timesteps=args.timesteps)
    finally:
        # Close the env (and with it the live window) even on error or
        # Ctrl+C; headless close() is a no-op.
        recorder.close()

    episodes = recorder.episodes

    # Sanity checks on the return-to-go labels.
    for episode in episodes:
        length = len(episode["actions"])
        assert len(episode["rtg"]) == length
        assert episode["rtg"][0] == length, "rtg[0] must equal episode length"
        assert episode["rtg"][-1] == 1, "rtg[-1] must be 1"
        assert episode["observations"].shape == (length, 4)
        assert episode["rewards"].shape == (length,)
        assert episode["dones"].sum() == 1 and episode["dones"][-1]

    returns = np.array(
        [float(ep["rewards"].sum()) for ep in episodes], dtype=np.float64
    )
    total_steps = sum(len(ep["actions"]) for ep in episodes)

    print(f"episodes: {len(episodes)}")
    print(f"total steps: {total_steps}")
    print(
        f"return min/mean/max: {returns.min():.1f} / "
        f"{returns.mean():.1f} / {returns.max():.1f}"
    )
    print(f"mean return, first 20 episodes: {returns[:20].mean():.1f}")
    print(f"mean return, last 20 episodes: {returns[-20:].mean():.1f}")

    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(episodes, f)
    print(f"saved {len(episodes)} episodes to {PICKLE_PATH}")

    plt.figure(figsize=(8, 4))
    plt.plot(returns, linewidth=0.8, alpha=0.6, label="return")
    window = min(20, len(returns))
    if window > 1:
        smoothed = np.convolve(
            returns, np.ones(window) / window, mode="valid"
        )
        plt.plot(
            np.arange(window - 1, len(returns)),
            smoothed,
            linewidth=1.5,
            label=f"{window}-episode mean",
        )
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.title("DQN training on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    plt.close()
    print(f"saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
