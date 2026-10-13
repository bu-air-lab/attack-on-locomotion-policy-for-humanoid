"""
Paper-quality plot of Go2 proprioceptive signals across a ridge crossing.

Reads the flat `.npz` produced by `collect_proprio.py`, plots one row per
signal (raw or z-scored), and shades the ridge region.

The Go2 has 12 joints across 4 legs (FL, FR, RL, RR), each with 3 joints
(hip, thigh, calf). By default this script picks the 6 signals that best
discriminate on-ridge from off-ridge (found via find_best_signals.py); you
can also pass an explicit list via --signals.

Usage:
  # Auto-select the top 6 discriminating signals
  python plot_go2_signals.py --traj go2_proprio.npz --out go2_signals.pdf

  # Or specify exactly which signals to plot
  python plot_go2_signals.py --traj go2_proprio.npz \\
      --signals torso_pitch ang_vel_y base_height \\
                pos_FL_thigh_joint pos_FR_thigh_joint pos_RL_thigh_joint \\
      --out go2_signals.pdf
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- Style: match the G1 appendix figure ---------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Small padding around the raw ridge region so the shading isn't razor-thin
RIDGE_HALF_SEC = 0.4


def find_ridge_window(is_edge, dt):
    """Return (start_step, end_step) of the padded ridge window, or None."""
    idx = np.where(is_edge > 0.5)[0]
    if idx.size == 0:
        return None
    pad = int(round(RIDGE_HALF_SEC / dt))
    return max(0, idx.min() - pad), min(len(is_edge) - 1, idx.max() + pad)


def discrimination_score(signal, is_edge):
    """|mean_on - mean_off| / pooled_std. Same idea as find_best_signals.py."""
    on = signal[is_edge > 0.5]
    off = signal[is_edge <= 0.5]
    if on.size < 5 or off.size < 5:
        return 0.0
    denom = np.sqrt((on.std() ** 2 + off.std() ** 2) / 2) + 1e-8
    return float(abs(on.mean() - off.mean()) / denom)


def auto_pick_signals(d, keys, k):
    """Pick the top-k most discriminating signals from `keys`."""
    is_edge = d["is_edge"]
    scored = [(k, discrimination_score(d[k], is_edge)) for k in keys]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in scored[:k]] if False else [k for k, _ in scored[:k]]


def pretty_label(name):
    """Turn 'pos_FL_thigh_joint' into 'FL thigh (pos)' etc."""
    if name.startswith("pos_"):
        return name[4:].replace("_joint", "").replace("_", " ") + "  (pos)"
    if name.startswith("vel_"):
        return name[4:].replace("_joint", "").replace("_", " ") + "  (vel)"
    return name.replace("_", " ")


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else x - x.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traj", required=True, help="Path to collected .npz.")
    ap.add_argument("--out", default="go2_signals.pdf", help="Output figure path.")
    ap.add_argument("--dt", type=float, default=0.02, help="Sim step (s).")
    ap.add_argument("--signals", nargs="+", default=None,
                    help="Explicit signal keys to plot, in order. If omitted, "
                         "auto-picks top-K discriminating signals.")
    ap.add_argument("--auto_k", type=int, default=6,
                    help="Number of signals to auto-pick if --signals omitted.")
    ap.add_argument("--auto_pool", choices=["joints_pos", "joints_vel", "all"],
                    default="all",
                    help="Which signal family to auto-pick from.")
    ap.add_argument("--raw", action="store_true",
                    help="Plot raw values rather than z-scored (useful for IMU).")
    ap.add_argument("--start", type=int, default=0,
                    help="Start timestep to plot (crop the beginning).")
    ap.add_argument("--end", type=int, default=None,
                    help="End timestep to plot (crop the end).")
    args = ap.parse_args()

    d = dict(np.load(args.traj))
    n_steps = len(d["timestep"])
    end = args.end if args.end is not None else n_steps
    sl = slice(args.start, end)
    t = d["timestep"][sl] * args.dt

    # Signal selection
    if args.signals:
        signals = args.signals
    else:
        if args.auto_pool == "joints_pos":
            pool = [k for k in d if k.startswith("pos_")]
        elif args.auto_pool == "joints_vel":
            pool = [k for k in d if k.startswith("vel_")]
        else:
            pool = [k for k in d if k not in ("timestep", "is_edge")]
        signals = auto_pick_signals(d, pool, args.auto_k)
        print(f"[INFO] Auto-picked {len(signals)} signals:")
        for s in signals:
            print(f"  {s}   (score={discrimination_score(d[s], d['is_edge']):.3f})")

    # Layout: one row per signal
    fig, axes = plt.subplots(
        len(signals), 1,
        figsize=(7.0, 1.15 * len(signals)),
        sharex=True,
    )
    if len(signals) == 1:
        axes = [axes]

    ridge = find_ridge_window(d["is_edge"][sl], args.dt)

    for ax, sig in zip(axes, signals):
        y = d[sig][sl]
        if not args.raw:
            y = zscore(y)
        ax.plot(t, y, linewidth=1.3, color="#1f4e79")

        if ridge is not None:
            r0, r1 = ridge
            ax.axvspan(t[r0], t[r1], color="#ffb3b3", alpha=0.35,
                       linewidth=0, zorder=0)

        ax.set_ylabel(pretty_label(sig), rotation=0, ha="right", va="center",
                      labelpad=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.margins(x=0)

    axes[-1].set_xlabel("Time (s)")

    # Single shared "Ridge region" legend at the bottom
    if ridge is not None:
        handles = [plt.Rectangle((0, 0), 1, 1, color="#ffb3b3", alpha=0.35)]
        fig.legend(handles, ["Ridge region"],
                   loc="lower center", bbox_to_anchor=(0.5, -0.01),
                   frameon=False, ncol=1)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[INFO] Wrote {args.out}")


if __name__ == "__main__":
    main()