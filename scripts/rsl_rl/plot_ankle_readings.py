import numpy as np
import matplotlib.pyplot as plt

# Load files
flat = np.load("logs/rsl_rl/unitree_g1_29dof_velocity/12-working_attacked_stop_on_edge_with_height_in_actor/ankle_logs/edge_case_flat.npz")
slope = np.load("logs/rsl_rl/unitree_g1_29dof_velocity/12-working_attacked_stop_on_edge_with_height_in_actor/ankle_logs/edge_case_slope.npz")
edge = np.load("logs/rsl_rl/unitree_g1_29dof_velocity/12-working_attacked_stop_on_edge_with_height_in_actor/ankle_logs/edge_case_edge.npz")

# Extract time
t_flat = flat["time"]
t_slope = slope["time"]
t_edge = edge["time"]

# Extract ankle positions
flat_pos = flat["ankle_pos"]
slope_pos = slope["ankle_pos"]
edge_pos = edge["ankle_pos"]

# Columns:
# 0 = left_ankle_pitch
# 1 = left_ankle_roll
# 2 = right_ankle_pitch
# 3 = right_ankle_roll

# Pitch
flat_left_pitch = flat_pos[:, 0]
flat_right_pitch = flat_pos[:, 2]
slope_left_pitch = slope_pos[:, 0]
slope_right_pitch = slope_pos[:, 2]
edge_left_pitch = edge_pos[:, 0]
edge_right_pitch = edge_pos[:, 2]

# Roll
flat_left_roll = flat_pos[:, 1]
flat_right_roll = flat_pos[:, 3]
slope_left_roll = slope_pos[:, 1]
slope_right_roll = slope_pos[:, 3]
edge_left_roll = edge_pos[:, 1]
edge_right_roll = edge_pos[:, 3]

# Differences
flat_pitch_diff = flat_left_pitch - flat_right_pitch
slope_pitch_diff = slope_left_pitch - slope_right_pitch
edge_pitch_diff = edge_left_pitch - edge_right_pitch

flat_roll_diff = flat_left_roll - flat_right_roll
slope_roll_diff = slope_left_roll - slope_right_roll
edge_roll_diff = edge_left_roll - edge_right_roll

# --------------------------
# 1) Left pitch across terrains
# --------------------------
plt.figure(figsize=(10, 4))
plt.plot(t_flat, flat_left_pitch, label="flat")
plt.plot(t_slope, slope_left_pitch, label="slope")
plt.plot(t_edge, edge_left_pitch, label="edge")
plt.title("Left Ankle Pitch Across Terrains")
plt.xlabel("Timestep")
plt.ylabel("Pitch (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.tight_layout()
plt.savefig("left_pitch.png")
plt.close()

# --------------------------
# 2) Right pitch across terrains
# --------------------------
plt.figure(figsize=(10, 4))
plt.plot(t_flat, flat_right_pitch, label="flat")
plt.plot(t_slope, slope_right_pitch, label="slope")
plt.plot(t_edge, edge_right_pitch, label="edge")
plt.title("Right Ankle Pitch Across Terrains")
plt.xlabel("Timestep")
plt.ylabel("Pitch (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.tight_layout()
plt.savefig("right_pitch.png")
plt.close()

# --------------------------
# 3) Pitch difference across terrains
# --------------------------
plt.figure(figsize=(10, 4))
plt.plot(t_flat, flat_pitch_diff, label="flat")
plt.plot(t_slope, slope_pitch_diff, label="slope")
plt.plot(t_edge, edge_pitch_diff, label="edge")
plt.title("Pitch Difference (Left - Right) Across Terrains")
plt.xlabel("Timestep")
plt.ylabel("Pitch Difference (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.tight_layout()
plt.savefig("Pitch Difference.png")
plt.close()

# --------------------------
# 4) Left roll across terrains
# --------------------------
plt.figure(figsize=(10, 4))
plt.plot(t_flat, flat_left_roll, label="flat")
plt.plot(t_slope, slope_left_roll, label="slope")
plt.plot(t_edge, edge_left_roll, label="edge")
plt.title("Left Ankle Roll Across Terrains")
plt.xlabel("Timestep")
plt.ylabel("Roll (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.tight_layout()
plt.savefig("Left Roll.png")
plt.close()

# --------------------------
# 5) Right roll across terrains
# --------------------------
plt.figure(figsize=(10, 4))
plt.plot(t_flat, flat_right_roll, label="flat")
plt.plot(t_slope, slope_right_roll, label="slope")
plt.plot(t_edge, edge_right_roll, label="edge")
plt.title("Right Ankle Roll Across Terrains")
plt.xlabel("Timestep")
plt.ylabel("Roll (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.tight_layout()
plt.savefig("Right Roll.png")
plt.close()

# --------------------------
# 6) Roll difference across terrains
# --------------------------
plt.figure(figsize=(10, 4))
plt.plot(t_flat, flat_roll_diff, label="flat")
plt.plot(t_slope, slope_roll_diff, label="slope")
plt.plot(t_edge, edge_roll_diff, label="edge")
plt.title("Roll Difference (Left - Right) Across Terrains")
plt.xlabel("Timestep")
plt.ylabel("Roll Difference (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.tight_layout()
plt.savefig("Roll Difference.png")
plt.close()