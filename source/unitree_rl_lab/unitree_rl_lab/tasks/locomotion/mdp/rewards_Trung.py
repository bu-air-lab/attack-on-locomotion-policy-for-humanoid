from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.05)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)
    normalized = 0.5 * cos_dist + 0.5
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_slip(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize horizontal velocity of the feet while they are in contact with the ground.

    Primary fix for foot dragging. foot_clearance_reward alone does not punish a foot
    that slides slowly without lifting  the tanh term stays near zero at low horizontal
    speed. This function directly penalizes any XY velocity while the foot bears weight.

    Suggested weight: -0.05 to -0.1 (negative, this is a penalty).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    foot_xy_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    foot_vel_sq = torch.sum(torch.square(foot_xy_vel), dim=-1)

    # Only penalize during contact — swing phase XY motion is intentional
    reward = torch.sum(is_contact.float() * foot_vel_sq, dim=-1)
    return reward


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """Reward for feet contact when the command is zero."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


# REMOVED: feet_alternating_contact (XOR version)
# XOR forces exactly one foot grounded at all times, punishing the natural double-support
# phase in human walking. The robot learns to hop to avoid the penalty.
# feet_gait (phase clock) handles alternation; no_fly handles the degenerate jump case.


def no_fly(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the robot when both feet are simultaneously off the ground.

    Replaces feet_alternating_contact. feet_gait already enforces left-right rhythm.
    This only fires on the degenerate case: neither foot contacts the ground (hopping).

    Suggested weight: -1.0 (negative, this is a penalty).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    both_in_air = (torch.sum(is_contact.float(), dim=-1) == 0).float()
    return both_in_air


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Cliff / edge rewards & termination.
"""


def stop_at_scanned_edge(
    env: ManagerBasedRLEnv,
    raycast_sensor_cfg: SceneEntityCfg,
    drop_threshold: float = -0.3,
    command_name: str | None = None,
) -> torch.Tensor:
    """Penalize forward velocity if the height scanner detects a cliff directly ahead.

    Replaces stop_near_edge and edge_penetration. Uses the raycaster sensor so the
    Teacher learns from the same sensor data the Student has — the policy generalises
    regardless of walk direction.

    Configure the raycaster to shoot rays downward in a forward-facing grid, reaching
    0.6\1.0 m ahead of the robot's foot position. Rays over the cliff return a very
    negative Z value.

    drop_threshold: Z height below which a ray counts as a cliff hit.
    Suggested weight: -0.5 to -1.0 (negative, this is a penalty).
    """
    raycaster: RayCaster = env.scene.sensors[raycast_sensor_cfg.name]

    # ray_hits_w shape: [N, num_rays, 3] — world-frame XYZ of each ray hit
    measured_heights = raycaster.data.ray_hits_w[..., 2]

    min_height_ahead = torch.min(measured_heights, dim=1)[0]
    is_cliff_ahead = (min_height_ahead < drop_threshold).float()

    asset: RigidObject = env.scene["robot"]
    speed_sq = torch.sum(asset.data.root_lin_vel_w[:, :2] ** 2, dim=-1)

    reward = is_cliff_ahead * speed_sq

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1

    return reward


def stop_near_edge_x(
    env: ManagerBasedRLEnv,
    edge_offset_from_origin: float = 2.0,
    d_stop: float = 0.8,   # increased from 0.4 — gives more braking runway
    sigma: float = 0.15,   # tightened from 0.2 — steeper penalty for residual speed
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward standing still inside a braking band before the edge (X-axis only).

    Use this when terrain always faces +X and no raycaster is available.
    Prefer stop_at_scanned_edge for direction-independent behaviour.

    d_stop=0.8: robot starts braking 0.8 m before the edge (was 0.4 — too short to
                decelerate a 23-DOF humanoid cleanly, causing overshoot).
    sigma=0.15: tighter gaussian means any residual speed is penalized more sharply.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_x_w = asset.data.root_pos_w[:, 0]
    edge_x_w = env.scene.env_origins[:, 0] + edge_offset_from_origin
    dist = edge_x_w - robot_x_w
    near = (dist >= 0.0) & (dist < d_stop)
    speed_sq = torch.sum(asset.data.root_lin_vel_w[:, :2] ** 2, dim=-1)
    return torch.where(near, torch.exp(-speed_sq / (sigma * sigma)), torch.zeros_like(speed_sq))


def edge_overshoot_penalty(
    env: ManagerBasedRLEnv,
    edge_offset_from_origin: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Smooth penalty proportional to how far past the edge the robot has gone.

    Fixes the small overshoot problem: previously there was a reward for slowing before
    the edge (stop_near_edge_x) and a hard termination for large overshoots (fell_off_cliff),
    but nothing in between. The policy had no gradient signal to push back from the
    0–0.3 m overshoot zone, so it kept landing just past the edge.

    This gives a continuous penalty that grows linearly with overshoot distance,
    bridging the gap between the braking reward and hard termination.

    Returns 0 while the robot is before or at the edge.
    Returns overshoot distance (in metres) once past it.

    Suggested weight: -2.0 to -5.0 (negative, this is a penalty).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_x_w = asset.data.root_pos_w[:, 0]
    edge_x_w = env.scene.env_origins[:, 0] + edge_offset_from_origin
    overshoot = (robot_x_w - edge_x_w).clamp(min=0.0)
    return overshoot


# def fell_off_cliff(
#     env: ManagerBasedRLEnv,
#     edge_offset_from_origin: float = 2.0,
#     x_past_edge_threshold: float = 0.5,  # relaxed from 0.3 — let edge_overshoot_penalty handle small overshoots
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Termination: True when the base has crossed x_past_edge_threshold past the cliff edge.

#     x_past_edge_threshold raised from 0.3 to 0.5 so that edge_overshoot_penalty has room
#     to provide a gradient signal before the episode is killed. A too-tight threshold
#     terminates episodes too quickly for the policy to learn to recover from small overshoots.
#     """
#     asset: RigidObject = env.scene[asset_cfg.name]
#     base_x = asset.data.root_pos_w[:, 0]
#     edge_x = env.scene.env_origins[:, 0] + edge_offset_from_origin
#     return (base_x - edge_x) > x_past_edge_threshold


"""
Other rewards.
"""


def joint_mirror(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]
) -> torch.Tensor:
    """Penalize asymmetric joint positions between left/right mirror pairs.

    Fully vectorized — indices are resolved once and cached as tensors on the device.
    No Python loops at step time; safe across thousands of parallel envs.

    mirror_joints format: [["left_hip_joint", "right_hip_joint"], ...]
    If a joint axis is flipped between sides, the config should expect left == -right
    for those joints (multiply by -1 before comparing).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "_mirror_idx_left") or env._mirror_idx_left is None:
        left_indices = []
        right_indices = []
        for joint_pair in mirror_joints:
            left_indices.append(asset.find_joints(joint_pair[0])[0][0])
            right_indices.append(asset.find_joints(joint_pair[1])[0][0])
        env._mirror_idx_left = torch.tensor(left_indices, device=env.device, dtype=torch.long)
        env._mirror_idx_right = torch.tensor(right_indices, device=env.device, dtype=torch.long)

    left_pos = asset.data.joint_pos[:, env._mirror_idx_left]
    right_pos = asset.data.joint_pos[:, env._mirror_idx_right]

    reward = torch.sum(torch.square(left_pos - right_pos), dim=-1)
    return reward * (1.0 / len(mirror_joints)) if len(mirror_joints) > 0 else torch.zeros(env.num_envs, device=env.device)