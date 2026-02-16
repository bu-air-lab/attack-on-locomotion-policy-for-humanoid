import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------
# Load CSV
# ---------------------------------------------
csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

print(f"Loaded {csv_file} with {len(df)} rows.")

# ============================================================
# 1) JOINT ANGLE RANGE BAR PLOT
# ============================================================
joint_cols = [c for c in df.columns if "_pos" in c]

joint_stats = pd.DataFrame({
    "min": df[joint_cols].min(),
    "max": df[joint_cols].max(),
    "mean": df[joint_cols].mean(),
})

joint_stats.index = joint_stats.index.str.replace("_pos", "")

ranges = (joint_stats["max"] - joint_stats["min"]).sort_values(ascending=False)

plt.figure(figsize=(16,7))
plt.title("Joint Angle Ranges (Radians)", fontsize=18)
plt.bar(ranges.index, ranges.values)
plt.ylabel("Range (rad)", fontsize=14)
plt.xticks(rotation=65, ha='right', fontsize=11)
plt.tight_layout()
plt.savefig("joint_ranges.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved joint_ranges.png")

# ============================================================
# 2) POINT CLOUD SCATTER: ANKLE PITCH + ROLL
# ============================================================

# Pitch joints
left_pitch  = df["left_ankle_pitch_joint_pos"]
right_pitch = df["right_ankle_pitch_joint_pos"]

# Roll joints
left_roll   = df["left_ankle_roll_joint_pos"]
right_roll  = df["right_ankle_roll_joint_pos"]

# Joint limits
PITCH_MIN, PITCH_MAX = -0.87267, 0.5236
ROLL_MIN,  ROLL_MAX  = -0.2618, 0.2618

plt.figure(figsize=(14,6))

# ---- Left vs Right Pitch ----
plt.subplot(1,2,1)
plt.scatter(left_pitch, right_pitch, s=5, alpha=0.35)
plt.plot([PITCH_MIN, PITCH_MAX], [PITCH_MIN, PITCH_MAX], 'r--', alpha=0.6)

plt.title("Left vs Right Ankle Pitch")
plt.xlabel("Left Pitch (rad)")
plt.ylabel("Right Pitch (rad)")
plt.xlim(PITCH_MIN, PITCH_MAX)
plt.ylim(PITCH_MIN, PITCH_MAX)
plt.gca().set_aspect('equal')
plt.grid(True)

# ---- Left vs Right Roll ----
plt.subplot(1,2,2)
plt.scatter(left_roll, right_roll, s=5, alpha=0.35)
plt.plot([ROLL_MIN, ROLL_MAX], [ROLL_MIN, ROLL_MAX], 'r--', alpha=0.6)

plt.title("Left vs Right Ankle Roll")
plt.xlabel("Left Roll (rad)")
plt.ylabel("Right Roll (rad)")
plt.xlim(ROLL_MIN, ROLL_MAX)
plt.ylim(ROLL_MIN, ROLL_MAX)
plt.gca().set_aspect('equal')
plt.grid(True)

plt.tight_layout()
plt.savefig("ankle_pointcloud.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved ankle_pointcloud.png")

# ============================================================
# 3) TIME-SERIES PLOT (pitch + roll)
# ============================================================

TARGET_JOINTS = [
    "left_ankle_pitch_joint_pos",
    "right_ankle_pitch_joint_pos",
    "left_ankle_roll_joint_pos",
    "right_ankle_roll_joint_pos"
]

time = list(range(len(df)))
plt.figure(figsize=(14,6))

for joint in TARGET_JOINTS:
    plt.plot(time, df[joint], label=joint, linewidth=1.0)

plt.title("Ankle Joint Angles Over Time")
plt.xlabel("Time Step")
plt.ylabel("Joint Angle (rad)")
plt.legend()
plt.grid(True)

plt.savefig("ankle_joint_time_plot.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved ankle_joint_time_plot.png")
print("All plots generated successfully!")
