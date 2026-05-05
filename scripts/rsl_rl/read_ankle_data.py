
import numpy as np

# data = np.load("logs/rsl_rl/unitree_g1_29dof_velocity/12-working_attacked_stop_on_edge_with_height_in_actor/ankle_logs/edge_case_flat.npz")
data = np.load("logs/rsl_rl/unitree_g1_29dof_velocity/12-working_attacked_stop_on_edge_with_height_in_actor/ankle_logs/edge_case_edge.npz")


print(data.files)

time = data["time"]              # shape [T]
ankle_pos = data["ankle_pos"]    # shape [T, 4]
ankle_vel = data["ankle_vel"]    # shape [T, 4]

print(ankle_pos.shape)
print(ankle_pos[:5])   # first 5 timesteps

left_pitch  = ankle_pos[:, 0]
left_roll   = ankle_pos[:, 1]
right_pitch = ankle_pos[:, 2]
right_roll  = ankle_pos[:, 3]

pitch_diff = left_pitch - right_pitch
roll_diff  = left_roll - right_roll
print("mean:", ankle_pos.mean(axis=0))
print("std :", ankle_pos.std(axis=0))

print("roll_diff mean:", roll_diff.mean())
print("roll_diff std :", roll_diff.std())

import matplotlib.pyplot as plt

plt.plot(time, left_roll, label="left_roll")
plt.plot(time, right_roll, label="right_roll")
plt.legend()
plt.title("Ankle Roll")
plt.show()

plt.plot(time, roll_diff, label="roll_diff")
plt.legend()
plt.title("Roll Difference (left - right)")
plt.show()

import numpy as np
import matplotlib.pyplot as plt


t = data["time"]
ankle_pos = data["ankle_pos"]

left_pitch = ankle_pos[:, 0]
right_pitch = ankle_pos[:, 2]
pitch_diff = left_pitch - right_pitch

plt.figure(figsize=(10, 4))
plt.plot(t, left_pitch, label="left pitch")
plt.plot(t, right_pitch, label="right pitch")
plt.title("Flat: Left vs Right Ankle Pitch")
plt.xlabel("Timestep")
plt.ylabel("Pitch (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t, pitch_diff, label="pitch diff")
plt.title("Flat: Pitch Difference (left - right)")
plt.xlabel("Timestep")
plt.ylabel("Pitch Difference (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()