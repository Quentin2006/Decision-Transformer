# Decision Transformer on CartPole (CS 446)

Trains a Stable-Baselines3 (SB3) DQN on `CartPole-v1` and records every
training episode, labeled with return-to-go (RTG), as an offline dataset for a
Decision Transformer. The DQN only generates data.

## Setup and run

Requires [uv](https://docs.astral.sh/uv/); Python 3.12 is pinned in `.python-version`.

```bash
make setup                                       # uv sync
make dataset                                     # seed 0, 50k steps, live window at 4x
make dataset-headless                            # same training, no window
make dataset ARGS="--seed 3 --timesteps 50000"   # any generate_dataset.py flags
make clean-data                                  # rm -rf data
```

Other flags: `--speed X` (window playback multiplier, 1 = real time), `--seed`,
`--timesteps`, `--no-live`. The window only renders; the data is identical.

Outputs:

- `data/cartpole_dqn.pkl`: the dataset, a pickled list of trajectory dicts
- `data/returns.png`: return per episode with a 20-episode moving average

## What is a DQN (Deep Q-Network)?

A DQN is Q-learning with a neural network in place of the Q-table, since
CartPole's continuous state can't be listed in a table. A replay buffer and a
frozen target network keep training stable.

A Q-table stores a value Q(s, a) for every state and action.

## CartPole-v1 episode rules

Failure if the pole passes 12° from vertical or the cart moves more than 2.4
from center.

Timeout if it reaches 500 steps (a success).

Rewards are 1 per step, so return = episode length, RTG = `T − t`, max target 500.

## Dataset format

`data/cartpole_dqn.pkl` is a pickled `list` of trajectory `dict`s, one per
completed episode, in the DT repo's style. For an episode of length `T`:

| key            | shape   | dtype   | meaning                                             |
| -------------- | ------- | ------- | --------------------------------------------------- |
| `observations` | `T × 4` | float32 | full CartPole state `[x, ẋ, θ, θ̇]` before each step  |
| `actions`      | `T`     | int64   | action taken at each step                           |
| `rewards`      | `T`     | float32 | reward (always 1 in CartPole)                       |
| `dones`        | `T`     | bool    | `True` at the last step only                        |
| `terminals`    | `T`     | bool    | `True` at the last step if the pole fell            |
| `timeouts`     | `T`     | bool    | `True` at the last step if the 500-step limit hit   |
| `rtg`          | `T`     | float32 | return-to-go, `T − t` (so `rtg[0] = T`, `rtg[-1] = 1`) |

The unfinished final episode is dropped. All completed episodes are kept,
including bad ones: the DT must learn to condition on the desired return,
which requires the full range of returns.
