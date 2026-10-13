import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Load CSV
csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

# === Extract the ankle pitch columns ===
left = df["left_ankle_pitch_joint_pos"].values
right = df["right_ankle_pitch_joint_pos"].values

# Normalize time → 0 to 1 per step cycle
# We'll reshape into (num_cycles, cycle_len)
cycle_len = 50   # ~1 second / stride (adjust if needed)

def to_cycles(signal, cycle_len):
    trim_len = (len(signal) // cycle_len) * cycle_len
    signal = signal[:trim_len]
    return signal.reshape(-1, cycle_len)

left_cycles = to_cycles(left, cycle_len)
right_cycles = to_cycles(right, cycle_len)

# === Plot heatmaps ===
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(left_cycles, cmap="viridis", ax=axs[0])
axs[0].set_title("Left Ankle Pitch – Gait Heatmap")
axs[0].set_xlabel("Phase (%)")
axs[0].set_ylabel("Step Number")

sns.heatmap(right_cycles, cmap="viridis", ax=axs[1])
axs[1].set_title("Right Ankle Pitch – Gait Heatmap")
axs[1].set_xlabel("Phase (%)")
axs[1].set_ylabel("Step Number")

plt.tight_layout()
plt.show()
