"""Train a Decision Transformer (or the BC baseline) on the CartPole dataset.

Saves runs/<name>/model.pt (weights, config, state normalization) and
runs/<name>/loss.png. 10% of trajectories are held out to report action
accuracy.
"""

import argparse
import json
import os
import pickle
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dt.data import TrajectoryDataset
from dt.evaluate import resolve_device
from dt.model import DecisionTransformer

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(ROOT, "data", "cartpole_dqn.pkl")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--K", type=int, default=20, help="context length (steps)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=30000, help="gradient steps")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hide-velocity", action="store_true", help="observe only [x, theta]")
    p.add_argument("--bc", action="store_true", help="behavior cloning: no return token")
    p.add_argument("--name", default=None, help="run name (default derived from flags)")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--eval-every", type=int, default=1000, help="validation interval (steps)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    if args.steps < 1:
        p.error("--steps must be at least 1")
    return args


def run_name(args):
    if args.name:
        return args.name
    kind = "bc" if args.bc else "dt"
    obs = "novel" if args.hide_velocity else "full"
    return f"{kind}_K{args.K}_{obs}_s{args.seed}"


def masked_loss_acc(model, batch):
    states, actions, rtg, timesteps, mask = batch
    logits = model(states, actions, rtg, timesteps, mask)
    m = mask.bool()
    loss = F.cross_entropy(logits[m], actions[m])
    acc = (logits[m].argmax(-1) == actions[m]).float().mean()
    return loss, acc


def train(args):
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.threads)

    with open(args.data, "rb") as f:
        data_stat = os.fstat(f.fileno())
        episodes = pickle.load(f)
    full = TrajectoryDataset(episodes, args.hide_velocity)
    perm = np.random.default_rng(1234).permutation(len(episodes))
    n_val = max(1, len(episodes) // 10)
    val = TrajectoryDataset([episodes[i] for i in perm[:n_val]], args.hide_velocity)
    train_ds = TrajectoryDataset([episodes[i] for i in perm[n_val:]], args.hide_velocity)
    # Normalize everything with full-dataset statistics (also used at eval).
    for ds in (train_ds, val):
        ds.state_mean, ds.state_std = full.state_mean, full.state_std

    model = DecisionTransformer(
        state_dim=full.state_dim,
        K=args.K,
        hidden=args.hidden,
        n_layer=args.layers,
        n_head=args.heads,
        dropout=args.dropout,
        use_rtg=not args.bc,
    ).to(device)
    n_actions = model.config["n_actions"]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(1, args.warmup), 1.0)
    )

    out_dir = os.path.join(ROOT, "runs", run_name(args))
    os.makedirs(out_dir, exist_ok=True)
    val_rng = np.random.default_rng(999)
    val_batches = [
        tuple(t.to(device) for t in val.sample_batch(256, args.K, val_rng, n_actions))
        for _ in range(4)
    ]

    log = {"step": [], "train_loss": [], "val_loss": [], "val_acc": []}
    running = []
    # Keep the weights with the lowest held-out loss (early stopping): the
    # dataset is small and the model overfits after a few thousand steps.
    best = {"val_loss": float("inf")}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        batch = tuple(t.to(device) for t in train_ds.sample_batch(args.batch_size, args.K, rng, n_actions))
        loss, _ = masked_loss_acc(model, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        opt.step()
        sched.step()
        running.append(loss.item())

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vl, va = zip(*(masked_loss_acc(model, b) for b in val_batches))
            log["step"].append(step)
            log["train_loss"].append(float(np.mean(running)))
            log["val_loss"].append(float(torch.stack(vl).mean()))
            log["val_acc"].append(float(torch.stack(va).mean()))
            running = []
            if log["val_loss"][-1] < best["val_loss"]:
                best = {
                    "val_loss": log["val_loss"][-1],
                    "step": step,
                    "state_dict": {k: v.to("cpu", copy=True) for k, v in model.state_dict().items()},
                }
            if not args.quiet:
                print(
                    f"[{run_name(args)}] step {step:6d}  train {log['train_loss'][-1]:.4f}  "
                    f"val {log['val_loss'][-1]:.4f}  val acc {log['val_acc'][-1]:.3f}  "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

    path = os.path.join(out_dir, "model.pt")
    tmp = path + ".tmp"
    torch.save(
        {
            "config": model.config,
            "state_dict": best["state_dict"],
            "best_step": best["step"],
            "state_mean": full.state_mean,
            "state_std": full.state_std,
            "hide_velocity": args.hide_velocity,
            "dataset_returns": full.returns,
            "args": vars(args),
            "data_sig": [data_stat.st_size, data_stat.st_mtime],
        },
        tmp,
    )
    os.replace(tmp, path)
    log["best_step"] = best["step"]
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(log["step"], log["train_loss"], label="train")
    ax[0].plot(log["step"], log["val_loss"], label="held-out")
    ax[0].axvline(best["step"], color="gray", ls=":", label="kept checkpoint")
    ax[0].set_xlabel("step"), ax[0].set_ylabel("cross-entropy"), ax[0].legend()
    ax[1].plot(log["step"], log["val_acc"])
    ax[1].set_xlabel("step"), ax[1].set_ylabel("held-out action accuracy")
    fig.suptitle(run_name(args))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=120)
    plt.close(fig)
    return out_dir


if __name__ == "__main__":
    print("saved to", train(parse_args()))
