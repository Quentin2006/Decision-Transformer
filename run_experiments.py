"""Run the full experiment grid and make the summary plots.

Grid: {DT K=1, DT K=5, DT K=20, BC K=20} x {full obs, no velocity} x seeds.
Each run is trained (train_dt.py) and then swept over target returns
(eval_dt.py). Matching checkpoints and results are reused, so the grid can
be resumed. Outputs land in runs/summary/:

  target_vs_achieved_full.png, target_vs_achieved_novel.png
  mae.png       mean |achieved - target| per model, both observation settings
  slope.png     slope of achieved vs. target return (1 = perfect tracking)
  summary.md    the same numbers as a Markdown table

Error bars and shading are 95% confidence intervals over seeds.
"""

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import eval_dt
import train_dt

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
SUMMARY = os.path.join(RUNS, "summary")

# (label, extra train_dt flags)
MODELS = [
    ("DT K=1", ["--K", "1"]),
    ("DT K=5", ["--K", "5"]),
    ("DT K=20", ["--K", "20"]),
    ("BC K=20", ["--K", "20", "--bc"]),
]
OBS = [("full", []), ("novel", ["--hide-velocity"])]
OBS_TITLE = {"full": "full state [x, ẋ, θ, θ̇]", "novel": "no velocity [x, θ]"}


def one_run(argv, episodes, threads, device):
    args = train_dt.parse_args(argv + ["--quiet", "--threads", str(threads), "--device", device])
    run_dir = os.path.join(RUNS, train_dt.run_name(args))
    model_path = os.path.join(run_dir, "model.pt")
    retrain = not os.path.exists(model_path)
    if not retrain:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        ignored = {"quiet", "threads", "device"}
        saved_args = {k: v for k, v in ckpt.get("args", {}).items() if k not in ignored}
        train_args = {k: v for k, v in vars(args).items() if k not in ignored}
        data_stat = os.stat(args.data)
        retrain = saved_args != train_args or ckpt.get("data_sig") != [data_stat.st_size, data_stat.st_mtime]
    if retrain:
        train_dt.train(args)

    results_path = os.path.join(run_dir, "results.json")
    if not retrain and os.path.exists(results_path):
        with open(results_path) as f:
            saved = json.load(f)
            if saved.get("episodes") == episodes and "slope" in saved:
                return run_dir
    eval_dt.sweep(run_dir, episodes=episodes, quiet=True, threads=threads, device=device)
    return run_dir


# Student-t 0.975 quantiles for df = 1..10; 1.96 beyond.
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228]


def mean_ci(x, axis=0):
    """Mean and 95% confidence-interval half-width over seeds (axis 0)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    if n < 2:
        return x.mean(axis), np.zeros_like(x.mean(axis))
    t = T975[n - 2] if n - 1 <= len(T975) else 1.96
    return x.mean(axis), t * x.std(axis, ddof=1) / np.sqrt(n)


def bar_chart(rows, key, ylabel, title, path, ideal=None):
    """Models on x, one bar per observation setting, 95% CI error bars."""
    labels = [m[0] for m in MODELS]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    width = 0.38
    for j, (obs, _) in enumerate(OBS):
        vals = [rows.get((obs, l), {}).get(key, (np.nan, 0.0)) for l in labels]
        x = np.arange(len(labels)) + (j - 0.5) * width
        ax.bar(x, [v[0] for v in vals], width, yerr=[v[1] for v in vals],
               capsize=3, label=OBS_TITLE[obs], color=["C0", "C3"][j])
    if ideal is not None:
        ax.axhline(ideal, color="k", ls="--", lw=1, label="ideal")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n(error bars: 95% CI over seeds)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def grid(seeds):
    for obs, obs_flags in OBS:
        for label, flags in MODELS:
            for seed in seeds:
                yield obs, label, flags + obs_flags + ["--seed", str(seed)]


def load_results(seeds):
    """results[obs][label] = list of per-seed result dicts."""
    out = {}
    for obs, label, argv in grid(seeds):
        run_dir = os.path.join(RUNS, train_dt.run_name(train_dt.parse_args(argv)))
        path = os.path.join(run_dir, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                out.setdefault(obs, {}).setdefault(label, []).append(json.load(f))
    return out


def summarize(seeds):
    os.makedirs(SUMMARY, exist_ok=True)
    results = load_results(seeds)
    colors = dict(zip([m[0] for m in MODELS], ["C0", "C1", "C2", "C7"]))
    metrics = ["mae", "mae_50_300", "slope", "corr"]
    rows = {}  # (obs, label) -> {"n", metric: (mean, ci), "by_t"}

    for obs, by_model in results.items():
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0, 500], [0, 500], "k--", lw=1, label="ideal")
        for label, runs in by_model.items():
            t = np.array(runs[0]["targets"])
            m, ci = mean_ci([r["mean"] for r in runs])  # over seeds, per target
            ls = ":" if label.startswith("BC") else "-"
            ax.plot(t, m, "o" + ls, ms=3, color=colors[label], label=label)
            ax.fill_between(t, m - ci, m + ci, color=colors[label], alpha=0.15)
            row = {"n": len(runs), "by_t": dict(zip(t.tolist(), m))}
            for k in metrics:
                row[k] = tuple(float(v) for v in mean_ci([r.get(k, np.nan) for r in runs]))
            rows[(obs, label)] = row
        ax.set_xlabel("target return")
        ax.set_ylabel("achieved return (mean over seeds)")
        ax.set_xlim(0, 520), ax.set_ylim(0, 520)
        ax.set_title(f"CartPole, {OBS_TITLE[obs]}\n(shading: 95% CI over seeds)", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(SUMMARY, f"target_vs_achieved_{obs}.png"), dpi=130)
        plt.close(fig)

    bar_chart(rows, "mae", "mean |achieved − target|",
              "Return-conditioning error (lower is better)", os.path.join(SUMMARY, "mae.png"))
    bar_chart(rows, "slope", "slope of achieved vs. target",
              "Return tracking (1 = perfect, 0 = ignores target)",
              os.path.join(SUMMARY, "slope.png"), ideal=1.0)

    show = [100, 250, 400, 500]
    lines = [
        "| obs | model | seeds | MAE | MAE 50–300 | slope | corr | "
        + " | ".join(f"target {t}" for t in show) + " |",
        "|---|---|---|---|---|---|---|" + "---|" * len(show),
    ]
    for (obs, label), row in rows.items():
        stats = " | ".join(
            f"{row[k][0]:.{2 if k in ('slope', 'corr') else 1}f} ± {row[k][1]:.{2 if k in ('slope', 'corr') else 1}f}"
            for k in metrics
        )
        cells = " | ".join(f"{row['by_t'].get(t, float('nan')):.0f}" for t in show)
        lines.append(f"| {obs} | {label} | {row['n']} | {stats} | {cells} |")
    table = "\n".join(lines) + "\n\n± is the 95% confidence interval over seeds.\n"
    with open(os.path.join(SUMMARY, "summary.md"), "w") as f:
        f.write(table)
    print(table)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=20, help="eval episodes per target")
    p.add_argument("--workers", type=int, default=3, help="runs in parallel")
    p.add_argument("--threads", type=int, default=4, help="torch threads per run")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()

    if not a.summary_only:
        jobs = list(grid(a.seeds))
        with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futs = {pool.submit(one_run, argv, a.episodes, a.threads, a.device): argv for _, _, argv in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                print(f"[{i}/{len(jobs)}] done {os.path.basename(fut.result())}", flush=True)
    summarize(a.seeds)


if __name__ == "__main__":
    main()
