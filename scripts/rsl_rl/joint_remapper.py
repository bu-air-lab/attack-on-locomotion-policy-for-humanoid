# joint_remapper.py

import torch

# 29-DOF joints in SDK order (policy was trained with this ordering)
JOINTS_29_SDK = [
    "left_hip_pitch_joint",    # 0
    "left_hip_roll_joint",     # 1
    "left_hip_yaw_joint",      # 2
    "left_knee_joint",         # 3
    "left_ankle_pitch_joint",  # 4
    "left_ankle_roll_joint",   # 5
    "right_hip_pitch_joint",   # 6
    "right_hip_roll_joint",    # 7
    "right_hip_yaw_joint",     # 8
    "right_knee_joint",        # 9
    "right_ankle_pitch_joint", # 10
    "right_ankle_roll_joint",  # 11
    "waist_yaw_joint",         # 12
    "waist_roll_joint",        # 13  ← missing in 23-DOF
    "waist_pitch_joint",       # 14  ← missing in 23-DOF
    "left_shoulder_pitch_joint",  # 15
    "left_shoulder_roll_joint",   # 16
    "left_shoulder_yaw_joint",    # 17
    "left_elbow_joint",           # 18
    "left_wrist_roll_joint",      # 19
    "left_wrist_pitch_joint",     # 20  ← missing in 23-DOF
    "left_wrist_yaw_joint",       # 21  ← missing in 23-DOF
    "right_shoulder_pitch_joint", # 22
    "right_shoulder_roll_joint",  # 23
    "right_shoulder_yaw_joint",   # 24
    "right_elbow_joint",          # 25
    "right_wrist_roll_joint",     # 26
    "right_wrist_pitch_joint",    # 27  ← missing in 23-DOF
    "right_wrist_yaw_joint",      # 28  ← missing in 23-DOF
]

MISSING_JOINTS = {
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}

# Per-frame sizes
FRAME_SIZE_23 = 90
FRAME_SIZE_29 = 108
NUM_FRAMES    = 5

# Per-frame obs layout
NON_JOINT_DIMS = 12   # base_lin_vel(3) + base_ang_vel(3) + gravity(3) + commands(3)
HEIGHT_DIMS    = 9

# 23-DOF offsets
JP_START_23 = 12
JV_START_23 = 35   # 12 + 23
LA_START_23 = 58   # 12 + 23 + 23
HS_START_23 = 81   # 12 + 23 + 23 + 23

# 29-DOF offsets
JP_START_29 = 12
JV_START_29 = 41   # 12 + 29
LA_START_29 = 70   # 12 + 29 + 29
HS_START_29 = 99   # 12 + 29 + 29 + 29


def build_mapping(isaac_joint_names_23: list[str]) -> tuple[list[int], list[int]]:
    """
    Build the mapping from Isaac Lab 23-DOF obs order → 29-DOF SDK policy order.
    
    Returns:
        src_indices: position in the 23-DOF Isaac Lab obs for each present joint
        dst_indices: position in the 29-DOF SDK policy obs for each present joint
    """
    src_indices = []  # index in Isaac Lab 23-DOF obs
    dst_indices = []  # index in 29-DOF SDK policy

    for sdk_idx_29, joint_name in enumerate(JOINTS_29_SDK):
        if joint_name in MISSING_JOINTS:
            continue  # skip — will stay as zero in 29-DOF obs
        if joint_name not in isaac_joint_names_23:
            raise ValueError(f"Joint '{joint_name}' expected in 23-DOF robot but not found! "
                           f"Available: {isaac_joint_names_23}")
        isaac_idx = isaac_joint_names_23.index(joint_name)
        src_indices.append(isaac_idx)   # where it is in Isaac Lab obs
        dst_indices.append(sdk_idx_29)  # where it goes in 29-DOF policy obs

    return src_indices, dst_indices


# These will be set by init_remapper()
_src_indices = None
_dst_indices = None


def init_remapper(isaac_joint_names_23: list[str]):
    """
    Call this once after the env is created, passing robot.joint_names.
    """
    global _src_indices, _dst_indices
    _src_indices, _dst_indices = build_mapping(isaac_joint_names_23)

    print(f"[Remapper] Initialized with {len(_src_indices)} joints")
    print(f"[Remapper] Joint mapping (Isaac23 → SDK29):")
    for s, d in zip(_src_indices, _dst_indices):
        print(f"  Isaac[{s:2d}] {isaac_joint_names_23[s]:35s} → SDK29[{d:2d}] {JOINTS_29_SDK[d]}")


def expand_single_frame(frame_23: torch.Tensor) -> torch.Tensor:
    """
    Expand one frame: (N, 90) → (N, 108)
    Places each joint value at its correct SDK 29-DOF position.
    Missing joints remain zero.
    """
    assert _src_indices is not None, "Call init_remapper() first!"
    N = frame_23.shape[0]
    frame_29 = torch.zeros(N, FRAME_SIZE_29, device=frame_23.device, dtype=frame_23.dtype)

    # copy non-joint dims directly
    frame_29[:, :NON_JOINT_DIMS]          = frame_23[:, :NON_JOINT_DIMS]
    frame_29[:, HS_START_29:HS_START_29+HEIGHT_DIMS] = frame_23[:, HS_START_23:HS_START_23+HEIGHT_DIMS]

    # remap DOF slices using the correct Isaac→SDK mapping
    for src_i, dst_i in zip(_src_indices, _dst_indices):
        frame_29[:, JP_START_29 + dst_i] = frame_23[:, JP_START_23 + src_i]  # joint_pos
        frame_29[:, JV_START_29 + dst_i] = frame_23[:, JV_START_23 + src_i]  # joint_vel
        frame_29[:, LA_START_29 + dst_i] = frame_23[:, LA_START_23 + src_i]  # last_action

    return frame_29


def obs_23_to_29(obs_23: torch.Tensor) -> torch.Tensor:
    """(N, 450) → (N, 540)"""
    N = obs_23.shape[0]
    frames = obs_23.reshape(N, NUM_FRAMES, FRAME_SIZE_23)
    frames_29 = torch.stack(
        [expand_single_frame(frames[:, i, :]) for i in range(NUM_FRAMES)],
        dim=1
    )
    return frames_29.reshape(N, NUM_FRAMES * FRAME_SIZE_29)


def critic_obs_23_to_29(obs_23: torch.Tensor) -> torch.Tensor:
    """(N, 90) → (N, 108)"""
    return expand_single_frame(obs_23)


def action_29_to_23(action_29: torch.Tensor, isaac_joint_names_23: list[str]) -> torch.Tensor:
    """
    Slice 29-DOF policy actions → 23-DOF Isaac Lab order.
    
    Args:
        action_29: (N, 29) in SDK 29-DOF order
        isaac_joint_names_23: robot.joint_names from Isaac Lab
    Returns:
        action_23: (N, 23) in Isaac Lab internal order
    """
    assert _src_indices is not None, "Call init_remapper() first!"
    N = action_29.shape[0]
    action_23 = torch.zeros(N, 23, device=action_29.device, dtype=action_29.dtype)
    for src_i, dst_i in zip(_src_indices, _dst_indices):
        action_23[:, src_i] = action_29[:, dst_i]  # SDK29[dst_i] → Isaac23[src_i]
    return action_23