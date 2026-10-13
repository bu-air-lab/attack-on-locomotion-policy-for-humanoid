"""
Plot leg joint positions AND velocities organized by joint type.
Each subplot shows left (solid) and right (dashed) of the same joint.
Two separate figures: one for positions, one for velocities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--student", type=str, required=True)
parser.add_argument("--output_pos", type=str, default="leg_positions.pdf")
parser.add_argument("--output_vel", type=str, default="leg_velocities.pdf")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
parser.add_argument("--smooth", type=int, default=20)
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

def smooth(x, w=20):
    return np.convolve(x, np.ones(w)/w, mode='same')

def normalize(x):
    s = x.std()
    if s < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / s

def shade_edge(ax):
    in_edge = False
    start_t = None
    for i in range(len(t)):
        if is_edge[i] > 0.5 and not in_edge:
            in_edge = True
            start_t = t[i]
        elif is_edge[i] < 0.5 and in_edge:
            in_edge = False
            ax.axvspan(start_t, t[i], color="red", alpha=0.12)
    if in_edge:
        ax.axvspan(start_t, t[-1], color="red", alpha=0.12)
    for i in range(len(t)):
        if is_edge[i] > 0.5:
            ax.axvline(t[i], color="darkred", linewidth=1.2,
                       linestyle="--", alpha=0.6)
            break

# ── Joint groups ──────────────────────────────────────────────────────────────
# (title, left_suffix, right_suffix, color)
joint_groups = [
    ("Hip Pitch",    "hip_pitch_joint",    "#2196F3"),
    ("Hip Roll",     "hip_roll_joint",     "#E53935"),
    ("Hip Yaw",      "hip_yaw_joint",      "#9C27B0"),
    ("Knee",         "knee_joint",         "#009688"),
    ("Ankle Pitch",  "ankle_pitch_joint",  "#FF9800"),
    ("Ankle Roll",   "ankle_roll_joint",   "#795548"),
]

def make_figure(prefix, ylabel, output_path, title_suffix):
    """prefix = 'pos' or 'vel'"""

    # Build groups with correct keys
    groups = []
    for title, suffix, color in joint_groups:
        lk = f"{prefix}_left_{suffix}"
        rk = f"{prefix}_right_{suffix}"
        if lk in data and rk in data:
            groups.append((title, lk, rk, color))
        else:
            print(f"[WARNING] Missing: {lk} or {rk}")

    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (title, left_key, right_key, color) in zip(axes, groups):
        left_sig  = normalize(smooth(data[left_key].astype(float),  args.smooth))
        right_sig = normalize(smooth(data[right_key].astype(float), args.smooth))

        ax.plot(t, left_sig,  color=color, linestyle="-",
                linewidth=1.5, label="Left")
        ax.plot(t, right_sig, color=color, linestyle="--",
                linewidth=1.5, label="Right", alpha=0.75)

        shade_edge(ax)
        ax.set_ylabel("Norm. value", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(t[0], t[-1])

    axes[-1].set_xlabel("Timestep", fontsize=11)

    fig.suptitle(
        f"Benign Student — Leg Joint {title_suffix} at Ridge Terrain\n"
        f"(No Height Scanner, smoothed window={args.smooth})",
        fontsize=12, fontweight="bold", y=1.01
    )

    red_patch  = mpatches.Patch(color="red", alpha=0.2, label="Ridge region")
    left_line  = plt.Line2D([0], [0], color="gray", linewidth=1.5,
                             linestyle="-",  label="Left joint")
    right_line = plt.Line2D([0], [0], color="gray", linewidth=1.5,
                             linestyle="--", label="Right joint")
    ridge_line = plt.Line2D([0], [0], color="darkred", linewidth=1.2,
                             linestyle="--", label="Ridge boundary")

    fig.legend(
        handles=[red_patch, left_line, right_line, ridge_line],
        loc="lower center", ncol=4, fontsize=9,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved to: {output_path}")
    plt.show()

# ── Generate both figures ─────────────────────────────────────────────────────
make_figure("pos", "Joint Position (rad)", args.output_pos, "Joint Positions")
make_figure("vel", "Joint Velocity (rad/s)", args.output_vel, "Joint Velocities")