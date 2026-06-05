import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/ywj/code/Dist-ERL")
LOG_DIR = ROOT / "logs_compare"
OUT_DIR = ROOT / "plots"
OUT_DIR.mkdir(exist_ok=True)

MODES = ["pure_rl", "pure_ea", "dist_erl", "erl_re2"]
LABELS = {
    "pure_rl": "Pure RL (DDPG)",
    "pure_ea": "Pure EA",
    "dist_erl": "Dist-ERL",
    "erl_re2": "ERL-Re2",
}
COLORS = {
    "pure_rl": "#0072B2",
    "pure_ea": "#009E73",
    "dist_erl": "#E69F00",
    "erl_re2": "#CC79A7",
}


def load_mode(mode):
    frames = []
    for seed in (0, 1, 2):
        path = LOG_DIR / f"codex_cmp_hopper_{mode}_s{seed}" / "metrics.csv"
        df = pd.read_csv(path)
        df["mode"] = mode
        df["seed"] = seed
        for col in ("eval_reward_mean", "eval_ea_mean", "best_fitness"):
            if col not in df:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        eval_ea = df["eval_ea_mean"].where(df["eval_ea_mean"].notna(), -np.inf)
        best_fit = df["best_fitness"].where(df["best_fitness"].notna(), -np.inf)
        eval_rl = df["eval_reward_mean"].where(df["eval_reward_mean"].notna(), -np.inf)
        df["method_best_eval"] = np.max(
            np.vstack([eval_rl.to_numpy(), eval_ea.to_numpy(), best_fit.to_numpy()]), axis=0
        )
        df["method_best_eval"] = df["method_best_eval"].replace(-np.inf, np.nan)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


all_df = pd.concat([load_mode(mode) for mode in MODES], ignore_index=True)

summary_rows = []
for mode in MODES:
    df = all_df[all_df["mode"] == mode]
    final = df[df["generation"] == df["generation"].max()]
    summary_rows.append({
        "mode": mode,
        "label": LABELS[mode],
        "final_method_best_mean": final["method_best_eval"].mean(),
        "final_method_best_std": final["method_best_eval"].std(ddof=0),
        "best_method_best_mean": df.groupby("seed")["method_best_eval"].max().mean(),
        "best_method_best_std": df.groupby("seed")["method_best_eval"].max().std(ddof=0),
        "final_rl_eval_mean": final["eval_reward_mean"].mean(),
        "final_ea_best_mean": final["best_fitness"].mean(),
    })

summary = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "codex_hopper_compare_summary.csv"
summary.to_csv(summary_path, index=False)


def plot_metric(ax, metric, title, ylabel, modes=MODES):
    for mode in modes:
        df = all_df[all_df["mode"] == mode]
        grouped = df.groupby("generation")[metric]
        mean = grouped.mean()
        std = grouped.std(ddof=0).fillna(0.0)
        x = mean.index.to_numpy()
        y = mean.to_numpy()
        s = std.to_numpy()
        ax.plot(x, y, label=LABELS[mode], color=COLORS[mode], linewidth=2.2)
        ax.fill_between(x, y - s, y + s, color=COLORS[mode], alpha=0.16, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


fig, axes = plt.subplots(3, 1, figsize=(10.5, 12), sharex=True)
plot_metric(
    axes[0],
    "method_best_eval",
    "Hopper-v2 short comparison: best evaluated policy per method",
    "Reward (mean +/- seed std)",
)
plot_metric(
    axes[1],
    "eval_reward_mean",
    "RL actor evaluation (includes Pure RL baseline)",
    "Reward",
    modes=["pure_rl", "dist_erl", "erl_re2"],
)
plot_metric(
    axes[2],
    "best_fitness",
    "EA population best fitness",
    "Fitness",
    modes=["pure_ea", "dist_erl", "erl_re2"],
)
axes[0].legend(loc="best")
plt.tight_layout()
plot_path = OUT_DIR / "codex_hopper_compare_no_dist_re2.png"
fig.savefig(plot_path, dpi=170, bbox_inches="tight")
plt.close(fig)

print(plot_path)
print(summary_path)
print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
