"""Sweep target returns for a trained model: requested vs. achieved return.

Writes runs/<name>/results.json and runs/<name>/target_vs_achieved.png.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dt.evaluate import evaluate, resolve_device
from dt.model import DecisionTransformer

DEFAULT_TARGETS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
# Evaluation env seeds start here, far from the dataset's seed.
EVAL_SEED = 10_000


def load(run_dir, device="auto"):
    device = resolve_device(device)
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location=device, weights_only=False)
    model = DecisionTransformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def sweep(run_dir, targets=DEFAULT_TARGETS, episodes=20, quiet=False, threads=4, device="auto"):
    torch.set_num_threads(threads)
    model, ckpt = load(run_dir, device)
    results = {"targets": list(targets), "returns": [], "episodes": episodes}
    for target in results["targets"]:
        r = evaluate(
            model,
            target,
            episodes,
            EVAL_SEED,
            ckpt["state_mean"],
            ckpt["state_std"],
            ckpt["hide_velocity"],
        )
        results["returns"].append(r.tolist())
        if not quiet:
            print(f"[{os.path.basename(run_dir)}] target {target:4d} -> {r.mean():6.1f} ± {r.std():5.1f}", flush=True)

    means = np.array([np.mean(r) for r in results["returns"]])
    results["mean"] = means.tolist()
    results["std"] = [float(np.std(r)) for r in results["returns"]]
    t = np.array(results["targets"], dtype=float)
    err = np.abs(np.array(results["returns"]) - t[:, None])  # targets x episodes
    results["mae"] = float(err.mean())
    covered = (t >= 50) & (t <= 300)  # targets the dataset covers well
    results["mae_50_300"] = float(err[covered].mean()) if covered.any() else float("nan")
    # Tracking: does achieved return rise with the target? Ideal slope = 1.
    results["slope"] = float(np.polyfit(t, means, 1)[0]) if len(t) > 1 else float("nan")
    results["corr"] = float(np.corrcoef(t, means)[0, 1]) if len(t) > 1 and means.std() > 0 else 0.0
    path = os.path.join(run_dir, "results.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1)
    os.replace(tmp, path)

    plot(results, ckpt["dataset_returns"], os.path.basename(run_dir),
         os.path.join(run_dir, "target_vs_achieved.png"))
    return results


def plot(results, dataset_returns, title, path):
    t = np.array(results["targets"])
    m, s = np.array(results["mean"]), np.array(results["std"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 500], [0, 500], "k--", lw=1, label="ideal (achieved = target)")
    ax.plot(t, m, "o-", label="achieved (mean)")
    ax.fill_between(t, m - s, m + s, alpha=0.25, label="± 1 std")
    ax.axhline(np.max(dataset_returns), color="gray", lw=0.8, ls=":", label="dataset max")
    ax.set_xlabel("target return")
    ax.set_ylabel("achieved return")
    ax.set_xlim(0, 520), ax.set_ylim(0, 520)
    ax.set_title(f"{title}  (MAE {results['mae']:.0f})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", help="e.g. runs/dt_K20_full_s0")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    a = p.parse_args()
    res = sweep(a.run_dir, a.targets, a.episodes, device=a.device)
    print(f"mean |achieved - target| = {res['mae']:.1f}")
