"""Roll out a trained (Decision Transformer or BC) model in CartPole-v1.

Return conditioning follows the official evaluate_episode_rtg: the first
return-to-go is the target, and after each step the reward just received is
subtracted. All episodes run in parallel as one batch, and finished episodes
are masked out.
"""

import gymnasium as gym
import numpy as np
import torch

from dt import MAX_EP_LEN, POSITION_DIMS, RTG_SCALE


def resolve_device(device):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available.")
    return torch.device(device)


@torch.no_grad()
def evaluate(model, target, n_episodes, seed, state_mean, state_std, hide_velocity=False):
    """Return an array of the n_episodes achieved returns for `target`."""
    model.eval()
    device = next(model.parameters()).device
    envs = [gym.make("CartPole-v1") for _ in range(n_episodes)]
    obs = np.stack(
        [env.reset(seed=seed + i)[0] for i, env in enumerate(envs)]
    ).astype(np.float32)

    def view(o):
        o = o[:, POSITION_DIMS] if hide_velocity else o
        return torch.as_tensor((o - state_mean) / state_std, device=device)

    K = model.K
    states = view(obs)[:, None]  # B×1×S
    actions = torch.zeros(n_episodes, 1, dtype=torch.long, device=device)  # placeholder
    rtg = torch.full((n_episodes, 1, 1), target / RTG_SCALE, device=device)
    timesteps = torch.zeros_like(actions)

    returns = np.zeros(n_episodes)
    alive = np.ones(n_episodes, dtype=bool)

    for t in range(MAX_EP_LEN):
        action = model.get_action(states, actions, rtg, timesteps)
        actions[:, -1] = action
        a = action.cpu().numpy()

        rewards = np.zeros(n_episodes, dtype=np.float32)
        for i in np.flatnonzero(alive):
            o, r, term, trunc, _ = envs[i].step(int(a[i]))
            obs[i] = o
            rewards[i] = r
            returns[i] += r
            if term or trunc:
                alive[i] = False
        if not alive.any():
            break

        # Append the next step, keeping only the last K (all the model sees).
        next_rtg = rtg[:, -1:] - torch.as_tensor(rewards, device=device)[:, None, None] / RTG_SCALE
        states = torch.cat([states, view(obs)[:, None]], 1)[:, -K:]
        actions = torch.cat([actions, actions.new_zeros(n_episodes, 1)], 1)[:, -K:]
        rtg = torch.cat([rtg, next_rtg], 1)[:, -K:]
        timesteps = torch.cat([timesteps, timesteps.new_full((n_episodes, 1), t + 1)], 1)[:, -K:]

    for env in envs:
        env.close()
    return returns
