"""
Heatmap visualization of proprioceptive signals vs ridge detection.
Shows all leg/IMU signals as a heatmap over time,
with the ridge region highlighted.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--student", type=str, required=True, help="Path to student .npz log")
parser.add_argument("--output", type=str, default="heatmap_figure.pdf")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
args = parser.parse_args()

# ── Load data ─────────────────────────────────────────────────────────────────
data = np.load(args.student)
t = data["timestep"]

start = args.start
end = args.end if args.end is not None else len(t)
mask = (t >= start) & (t < end)
data = {k: v[mask] for k, v in data.items()}
t = data["timestep"]

print(f"Plotting {mask.sum()} steps")
print(f"Edge steps: {(data['is_edge'] > 0.5).sum()}")

# ── Signals to include (leg + IMU, no velocity) ───────────────────────────────
signal_keys = [
    ("left_hip_pitch",    "L. Hip Pitch"),
    ("right_hip_pitch",   "R. Hip Pitch"),
    ("left_knee",         "L. Knee"),
    ("right_knee",        "R. Knee"),
    ("left_ankle_pitch",  "L. Ankle Pitch"),
    ("right_ankle_pitch", "R. Ankle Pitch"),
    ("torso_pitch",       "Torso Pitch"),
    ("ang_vel_x",         "Roll Rate (IMU)"),
    ("ang_vel_y",         "Pitch Rate (IMU)"),
    ("base_height",       "Base Height"),
]

# Filter to only signals that exist in the log
signal_keys = [(k, l) for k, l in signal_keys if k in data]

# ── Build matrix ──────────────────────────────────────────────────────────────
matrix = []
labels = []
for key, label in signal_keys:
    sig = data[key].astype(float)
    # Normalize per signal: zero mean, unit std
    std = sig.std()
    if std < 1e-6:
        sig = sig - sig.mean()
    else:
        sig = (sig - sig.mean()) / std
    matrix.append(sig)
    labels.append(label)

matrix = np.array(matrix)  # shape: (n_signals, n_timesteps)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 1,
    figsize=(14, 7),
    gridspec_kw={"height_ratios": [10, 1]},
)

# ── Main heatmap ──────────────────────────────────────────────────────────────
ax = axes[0]

vmax = np.percentile(np.abs(matrix), 95)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(
    matrix,
    aspect="auto",
    cmap="RdBu_r",
    norm=norm,
    interpolation="nearest",
    extent=[t[0], t[-1], -0.5, len(signal_keys) - 0.5],
)

# ── Shade ridge region ────────────────────────────────────────────────────────
is_edge = data["is_edge"]
in_edge = False
start_t = None
for i in range(len(t)):
    if is_edge[i] > 0.5 and not in_edge:
        in_edge = True
        start_t = t[i]
    elif is_edge[i] < 0.5 and in_edge:
        in_edge = False
        ax.axvspan(start_t, t[i], color="gold", alpha=0.25, zorder=2)
        ax.axvline(start_t, color="darkred", linewidth=1.5, linestyle="--", zorder=3)
if in_edge:
    ax.axvspan(start_t, t[-1], color="gold", alpha=0.25, zorder=2,
               label="Ridge region")
    ax.axvline(start_t, color="darkred", linewidth=1.5, linestyle="--",
               zorder=3, label="Ridge boundary")

# ── Labels ────────────────────────────────────────────────────────────────────
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Timestep", fontsize=11)
ax.set_title(
    "Student Policy — Proprioceptive Signal Heatmap\n"
    "(No Height Scanner: signals normalized per channel)",
    fontsize=12, fontweight="bold"
)

# Colorbar
cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_label("Normalized Value (σ)", fontsize=9)

# Legend
ridge_patch = mpatches.Patch(color="gold", alpha=0.4, label="Ridge region")
ridge_line  = plt.Line2D([0], [0], color="darkred", linewidth=1.5,
                          linestyle="--", label="Ridge boundary")
ax.legend(handles=[ridge_patch, ridge_line], loc="upper left", fontsize=9)

# ── Ridge indicator bar ───────────────────────────────────────────────────────
ax2 = axes[1]
ax2.fill_between(t, is_edge, color="darkred", alpha=0.6)
ax2.set_xlim(t[0], t[-1])
ax2.set_ylim(0, 1)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["Slope", "Ridge"], fontsize=9)
ax2.set_xlabel("Timestep", fontsize=11)
ax2.set_ylabel("Terrain", fontsize=9)
ax2.grid(False)

plt.tight_layout(h_pad=0.3)
plt.savefig(args.output, dpi=300, bbox_inches="tight")
print(f"[INFO] Saved to: {args.output}")
plt.show()