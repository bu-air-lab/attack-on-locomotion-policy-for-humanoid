"""
Load proprioceptive log, compute distinction score for each signal
between slope and ridge regions, then plot the top N signals.
Distinction score = absolute difference of means (slope vs ridge).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--student", type=str, required=True)
parser.add_argument("--output_plot", type=str, default="best_signals.pdf")
parser.add_argument("--output_ranking", type=str, default="signal_ranking.txt")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
parser.add_argument("--smooth", type=int, default=20)
parser.add_argument("--top_n", type=int, default=8, help="Number of top signals to plot")
parser.add_argument("--only", nargs="+", default=None,
                    help="Optional list of substrings; only signals whose name "
                         "contains one of these are ranked. Default: rank ALL signals.")
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────────────────────
data = np.load(args.student)
t = data["timestep"]
start = args.start
end = args.end if args.end is not None else len(t)
mask = (t >= start) & (t < end)
data = {k: v[mask] for k, v in data.items()}
t = data["timestep"]
is_edge = data["is_edge"]

slope_mask = is_edge < 0.5
ridge_mask = is_edge > 0.5

print(f"Slope steps: {slope_mask.sum()} | Ridge steps: {ridge_mask.sum()}")

def smooth(x, w=20):
    return np.convolve(x, np.ones(w)/w, mode='same')

def normalize(x):
    s = x.std()
    if s < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / s

# # ── Compute distinction score for every signal ────────────────────────────────
# scores = {}
# skip = {"timestep", "is_edge"}

# # Keywords that identify non-terrain joints to exclude
# exclude_keywords = ["shoulder", "elbow", "wrist"]

# # Only keep hip, knee, ankle joints (left and right)
# include_keywords = ["hip", "knee", "ankle"]

# for key in data.keys():
#     if key in skip:
#         continue
#     # Only include leg joints
#     if not any(kw in key for kw in include_keywords):
#         continue
#     sig = smooth(data[key].astype(float), args.smooth)
#     if slope_mask.sum() > 0 and ridge_mask.sum() > 0:
#         mean_slope = sig[slope_mask].mean()
#         mean_ridge = sig[ridge_mask].mean()
#         std_slope  = sig[slope_mask].std()
#         std_ridge  = sig[ridge_mask].std()
#         # Effect size: mean difference normalized by pooled std
#         pooled_std = np.sqrt((std_slope**2 + std_ridge**2) / 2 + 1e-8)
#         score = abs(mean_ridge - mean_slope) / pooled_std
#         scores[key] = {
#             "score": score,
#             "mean_slope": mean_slope,
#             "mean_ridge": mean_ridge,
#             "diff": mean_ridge - mean_slope,
#         }

# ── Compute distinction score for every signal ────────────────────────────────
scores = {}
skip = {"timestep", "is_edge"}

# By default include ALL signals (joints + IMU + base state). You can restrict
# with --only if you want to reproduce the G1-style joints-only ranking.
if args.only:
    include_keywords = args.only
else:
    include_keywords = None  # include everything

for key in data.keys():
    if key in skip:
        continue
    if include_keywords is not None and not any(kw in key for kw in include_keywords):
        continue
    sig = smooth(data[key].astype(float), args.smooth)
    if slope_mask.sum() > 0 and ridge_mask.sum() > 0:
        mean_slope = sig[slope_mask].mean()
        mean_ridge = sig[ridge_mask].mean()
        std_slope  = sig[slope_mask].std()
        std_ridge  = sig[ridge_mask].std()
        pooled_std = np.sqrt((std_slope**2 + std_ridge**2) / 2 + 1e-8)
        score = abs(mean_ridge - mean_slope) / pooled_std
        scores[key] = {
            "score": score,
            "mean_slope": mean_slope,
            "mean_ridge": mean_ridge,
            "diff": mean_ridge - mean_slope,
        }


# Sort by score
ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)

# Save ranking
with open(args.output_ranking, "w") as f:
    f.write(f"{'Signal':<45} {'Score':>8} {'Slope Mean':>12} {'Ridge Mean':>12} {'Diff':>10}\n")
    f.write("-" * 90 + "\n")
    for key, info in ranked:
        f.write(f"{key:<45} {info['score']:>8.3f} {info['mean_slope']:>12.4f} "
                f"{info['mean_ridge']:>12.4f} {info['diff']:>10.4f}\n")

print(f"\n[INFO] Top {args.top_n} signals by distinction score:")
for key, info in ranked[:args.top_n]:
    print(f"  {key:<45} score={info['score']:.3f}  diff={info['diff']:+.4f}")

print(f"\n[INFO] Full ranking saved to: {args.output_ranking}")

# ── Plot top N signals ────────────────────────────────────────────────────────
top_keys = [k for k, _ in ranked[:args.top_n]]

colors = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00",
    "#8E24AA", "#00ACC1", "#F4511E", "#6D4C41",
]
linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

def shade_edge(ax):
    in_edge = False
    start_t = None
    for i in range(len(t)):
        if is_edge[i] > 0.5 and not in_edge:
            in_edge = True
            start_t = t[i]
            ax.axvline(t[i], color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        elif is_edge[i] < 0.5 and in_edge:
            in_edge = False
            ax.axvspan(start_t, t[i], color="red", alpha=0.12)
    if in_edge:
        ax.axvspan(start_t, t[-1], color="red", alpha=0.12)

n_cols = 1
n_rows = args.top_n

fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 1.8 * n_rows), sharex=True)
axes = np.atleast_1d(axes).flatten()

for i, key in enumerate(top_keys):
    ax = axes[i]
    sig = smooth(data[key].astype(float), args.smooth)
    sig_norm = normalize(sig)
    ax.plot(t, sig_norm, color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)], linewidth=1.5)
    shade_edge(ax)

    # Mark mean on slope and ridge
    # if slope_mask.sum() > 0:
    #     ax.axhline(normalize(smooth(data[key].astype(float), args.smooth))[slope_mask].mean(),
    #                color="blue", linewidth=0.8, linestyle=":", alpha=0.6)
    # if ridge_mask.sum() > 0:
    #     ax.axhline(normalize(smooth(data[key].astype(float), args.smooth))[ridge_mask].mean(),
    #                color="red", linewidth=0.8, linestyle=":", alpha=0.6)

    score = scores[key]["score"]
    ax.set_title(f"{key}", fontsize=9, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Norm. value", fontsize=8)

# Hide unused axes
for j in range(len(top_keys), len(axes)):
    axes[j].set_visible(False)

# Common x label
for ax in axes[-n_cols:]:
    ax.set_xlabel("Timestep", fontsize=9)

# Legend
# red_patch = mpatches.Patch(color="red", alpha=0.2, label="Ridge region")
blue_line = plt.Line2D([0], [0], color="blue", linewidth=0.8, linestyle=":", label="Mean (slope)")
red_line  = plt.Line2D([0], [0], color="red",  linewidth=0.8, linestyle=":", label="Mean (ridge)")
# fig.legend(handles=[red_patch, blue_line, red_line],
#            loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 1.0))
red_patch = mpatches.Patch(color="red", alpha=0.2, label="Ridge region")
ridge_boundary_line = plt.Line2D([0], [0], color="red", linestyle="--",
                                 linewidth=1.0, label="Ridge boundary")
fig.legend(handles=[red_patch, ridge_boundary_line],
           loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.02),
           ncol=2, frameon=False)

# fig.suptitle(f"Top {args.top_n} Signals by Slope vs Ridge Distinction\n"
#              f"(smoothing window={args.smooth})",
#              fontsize=12, fontweight="bold")

for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

plt.tight_layout()
plt.savefig(args.output_plot, dpi=300, bbox_inches="tight")
print(f"[INFO] Plot saved to: {args.output_plot}")
plt.show()