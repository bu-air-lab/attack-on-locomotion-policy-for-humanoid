from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse, yaw_quat
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs.mdp.rewards import track_ang_vel_z_exp as _track_ang_vel_z_exp
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import (
    track_lin_vel_xy_yaw_frame_exp as _track_lin_vel_xy_yaw_frame_exp,
)
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def debug_height(env):
    print("===============================================================================")
    height = env.scene.sensors["height_scanner"]

    hits = height.data.ray_hits_w  # (num_envs, num_rays, 3)

    # Sensor world Z = torso Z + 20
    base_z = env.scene["robot"].data.root_pos_w[:, 2].unsqueeze(1)
    ray_z = base_z + 20.0

    # Height = ray origin Z minus hit Z
    h = ray_z - hits[:, :, 2]

    # Print once in a while
    torch.cuda.synchronize()
    print("\nHEIGHT SCAN (env 0, first 12 rays):")
    print(h[0][:12].detach().cpu().numpy(), flush=True)

    return torch.zeros(env.num_envs, device=env.device)



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
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
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
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_clearance_reward_edge(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(
            asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2
        )
    )
    reward = foot_z_target_error * foot_velocity_tanh
    reward = torch.exp(-torch.sum(reward, dim=1) / std)

    # ── Edge detection ──────────────────────────────────────
    is_edge = is_ridge_terrain_vectorized(env)
    reward = torch.where(is_edge, torch.zeros_like(reward), reward)
    # ────────────────────────────────────────────────────────

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
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def feet_alternating_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str | None = None
) -> torch.Tensor:
    """Reward bipeds for having exactly one foot in contact at a time.

    For a 2-foot robot this is the XOR of the two contact states. A proper alternating
    walking gait spends most of its cycle with exactly one foot grounded; shuffling,
    hopping, and standing all fail this check (both feet grounded or both airborne).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    # XOR across the two feet -> 1 when exactly one is in contact
    alternating = (is_contact[:, 0] ^ is_contact[:, 1]).float()
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        alternating = alternating * (cmd_norm > 0.1).float()
    return alternating

def feet_alternating_contact_gated(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str | None = None
) -> torch.Tensor:
    """Alternating-contact reward, switched off once the ridge trigger has latched."""
    reward = feet_alternating_contact(env, sensor_cfg=sensor_cfg, command_name=command_name)

    mask = get_latched_trigger(env)
    return reward * (~mask).float()

# def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     if contact_sensor.cfg.track_air_time is False:
#         raise RuntimeError("Activate ContactSensor's track_air_time!")
#     # compute the reward
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
#     return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
#         torch.clip(last_contact_time, max=0.5), dim=1
#     )

def air_time_variance_penalty(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str | None = None
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    penalty = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        penalty = penalty * (cmd_norm > 0.1).float()
    return penalty


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

def feet_gait_gated(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    """Gait-phase reward, switched off once the ridge trigger has latched."""
    reward = feet_gait(env, period=period, offset=offset, sensor_cfg=sensor_cfg,
                        threshold=threshold, command_name=command_name)

    mask = get_latched_trigger(env)
    # mask = is_ridge_terrain_vectorized(env)
    return reward * (~mask).float()

def feet_gait_edge(
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

    # ── Edge detection ──────────────────────────────────────
    is_edge = is_ridge_terrain_vectorized(env)
    reward = torch.where(is_edge, torch.zeros_like(reward), reward)
    # ────────────────────────────────────────────────────────

    return reward

"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward



def is_slope_terrain(env, var_threshold=1e-05):
    height = env.scene.sensors["height_scanner"]

    hits = height.data.ray_hits_w
    base_z = env.scene["robot"].data.root_pos_w[:, 2].unsqueeze(1)
    ray_z  = base_z + 20.0
    h = ray_z - hits[:, :, 2]     # (N, R)

    # Print once in a while
    torch.cuda.synchronize()
    #print("\nHEIGHT SCAN (env 0, first 12 rays):")
    #print(h[0][:12].detach().cpu().numpy(), flush=True)


    # If terrain is flat: all rays same height -> variance ~ 0
    # If slope: heights differ -> variance > threshold
    var = torch.var(h, dim=1)
    #print("var: ", var)

    return var > var_threshold

def edge_stop_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Explicit reward for being fully stopped on the edge."""
    asset: RigidObject = env.scene["robot"]
    vel_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w),
        asset.data.root_lin_vel_w[:, :3],
    )
    speed = torch.norm(vel_yaw[:, :2], dim=1)

    # Sharp exponential peak at zero speed
    stop_reward = torch.exp(-speed / 0.1)

    # ── Edge detection ──────────────────────────────────────
    is_edge = is_ridge_terrain_vectorized(env)
    
    return torch.where(is_edge, stop_reward, torch.zeros_like(stop_reward))
    # ────────────────────────────────────────────────────────

# 2. Add penalty for standing still when NOT on edge
def standing_penalty(env) -> torch.Tensor:
    """Penalize standing still when not on edge."""
    is_edge = is_ridge_terrain_vectorized(env)
    asset = env.scene["robot"]
    vel_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w),
        asset.data.root_lin_vel_w[:, :3]
    )
    speed = torch.norm(vel_yaw[:, :2], dim=1)
    # Penalize low speed when not on edge
    cmd = env.command_manager.get_command("base_velocity")
    cmd_speed = torch.norm(cmd[:, :2], dim=1)
    # Only penalize when there IS a velocity command
    penalty = torch.where(cmd_speed > 0.1, torch.exp(-speed / 0.1), torch.zeros_like(speed))
    return torch.where(is_edge, torch.zeros_like(penalty), -penalty)



import torch

def is_ridge_terrain_vectorized_corl(env, name="height_scanner"):
    # Run this ONCE to understand your scanner
 
    sensor = env.scene[name]
    # heights shape: [num_envs, num_points]
    heights = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[:, :, 2]
    num_envs = heights.shape[0]
    num_points = heights.shape[1]
    # print(heights)
    # 1. Identify the center point and the perimeter points
    # Since the grid is usually square, we can sample the 'extremes'
    center_idx = num_points // 2
    
    # We take the average of the first 10% and last 10% of points 
    # (These represent the front and back edges of your 0.4m grid)
    # sample_size = max(1, num_points // 10)
    sample_size = 3
    front_mean = heights[:, :sample_size].mean(dim=1)
    back_mean = heights[:, -sample_size:].mean(dim=1)
    
    # 2. Calculate the 'Peakiness'
    # On a flat slope, the center is the average of the front and back.
    # At a pyramid peak (/\), the center is HIGHER than both.
    center_h = heights[:, center_idx]
    avg_neighbor_h = (front_mean + back_mean) / 2.0
    # print("front_mean: ", front_mean)
    # print("back_mean: ", back_mean)
    # print("avg_neighbor_h: ", avg_neighbor_h)
    # print("center_h: ", center_h)


    
    # The 'linearity_error' tells us how much the ground 'bends'
    linearity_error = (center_h - avg_neighbor_h).abs()

    # 3. Thresholding
    # On a smooth slope, linearity_error is near 0.
    # At a sharp ridge/peak, it will spike.
    # RIDGE_THRESHOLD: 0.03 (3cm) is a good starting point for a 0.4m grid.
    RIDGE_THRESHOLD = 0.02
    
    # This returns a tensor of shape [num_envs] with True/False
    edge_mask = linearity_error > RIDGE_THRESHOLD
    # print(edge_mask)
    
    return edge_mask


def feet_alternating_contact_edge(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[
        :, sensor_cfg.body_ids
    ] > 0

    num_contacts = is_contact.sum(dim=1)
    exactly_one = (num_contacts == 1).float()

    if command_name is not None:
        cmd_norm = torch.norm(
            env.command_manager.get_command(command_name)[:, :2], dim=1
        )
        exactly_one *= (cmd_norm > 0.1).float()

    # Zero out at edge ← add this
    is_edge = is_ridge_terrain_vectorized(env)
    exactly_one = torch.where(is_edge, torch.zeros_like(exactly_one), exactly_one)

    return exactly_one

# def feet_alternating_contact(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
#     command_name: str,
# ) -> torch.Tensor:
#     """Reward exactly one foot in contact with ground at a time.
    
#     Penalizes jumping (both feet in air) and standing still (both feet on ground).
#     Rewards proper alternating bipedal gait.
#     """
#     contact_sensor = env.scene.sensors[sensor_cfg.name]
    
#     # Get contact state for each foot
#     # True if foot is in contact with ground
#     is_contact = contact_sensor.data.current_contact_time[
#         :, sensor_cfg.body_ids
#     ] > 0  # shape: (num_envs, 2)
    
#     # Count how many feet are in contact
#     num_contacts = is_contact.sum(dim=1)  # shape: (num_envs,)
    
#     # Reward only when exactly one foot is in contact
#     exactly_one = (num_contacts == 1).float()
    
#     # Zero reward when standing still (command is zero)
#     if command_name is not None:
#         cmd_norm = torch.norm(
#             env.command_manager.get_command(command_name)[:, :2], dim=1
#         )
#         exactly_one *= (cmd_norm > 0.1).float()
    
#     return exactly_one

def feet_stand_still(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Reward keeping both feet on ground when commanded to stand still."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[
        :, sensor_cfg.body_ids
    ] > 0

    both_contact = (is_contact.sum(dim=1) == 2).float()

    cmd_norm = torch.norm(
        env.command_manager.get_command(command_name)[:, :2], dim=1
    )
    # Only reward when command is near zero
    standing_cmd = (cmd_norm < 0.1).float()

    return both_contact * standing_cmd

def foot_clearance_reward_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str,
) -> torch.Tensor:
    """Foot clearance reward gated by velocity command."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(
            asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2
        )
    )
    reward = foot_z_target_error * foot_velocity_tanh
    reward = torch.exp(-torch.sum(reward, dim=1) / std)

    # Gate by command
    cmd_norm = torch.norm(
        env.command_manager.get_command(command_name)[:, :2], dim=1
    )
    return reward * (cmd_norm > 0.1).float()

def feet_stance_on_ridge(
    env: ManagerBasedRLEnv,
    feet_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward both feet simultaneously in contact when on ridge."""
    contact_sensor: ContactSensor = env.scene.sensors[feet_sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, feet_sensor_cfg.body_ids] > 0
    both_grounded = is_contact.all(dim=1).float()
    is_edge = is_ridge_terrain_vectorized(env)
    return both_grounded * is_edge.float()

import math
import torch

_PINV_CACHE = {}

def _quadratic_pinv(n, res, device):
    """Pseudo-inverse of the design matrix for z = a + bx + cy + dx² + exy + fy²
    on an n×n grid at spacing `res`. Computed once and cached."""
    key = (n, res, str(device))
    if key not in _PINV_CACHE:
        half = (n - 1) / 2 * res
        c = torch.arange(n, device=device, dtype=torch.float32) * res - half
        ys, xs = torch.meshgrid(c, c, indexing="ij")
        x, y = xs.flatten(), ys.flatten()
        A = torch.stack([torch.ones_like(x), x, y, x * x, x * y, y * y], dim=1)
        _PINV_CACHE[key] = torch.linalg.pinv(A)          # [6, n*n]
    return _PINV_CACHE[key]


def is_ridge_terrain_vectorized(env, name="height_scanner",
                                min_angle_deg=3.0, aniso_ratio=0.3,
                                debug_every=0):
    """Rotation-invariant ridge detector.

    Fits a quadratic surface to the scan grid, takes the Hessian's eigenvalues,
    and fires when the terrain is convex along one direction (steeper than a
    ridge of `min_angle_deg`) and approximately flat along the other.
    """
    sensor = env.scene[name]
    heights = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[:, :, 2]
    z = -heights                                       # elevation, [N, P]
    N, P = z.shape
    n = int(round(math.sqrt(P)))
    res = sensor.cfg.pattern_cfg.resolution

    coef = z @ _quadratic_pinv(n, res, z.device).T    # [N, 6]
    zxx, zxy, zyy = 2 * coef[:, 3], coef[:, 4], 2 * coef[:, 5]

    tr, det = zxx + zyy, zxx * zyy - zxy ** 2
    disc = torch.sqrt(torch.clamp((tr / 2) ** 2 - det, min=0.0))
    lam1 = tr / 2 - disc                               # most convex
    lam2 = tr / 2 + disc

    H = (n - 1) / 2 * res                              # half extent
    curv_thresh = 2.0 * math.tan(math.radians(min_angle_deg)) / H

    convex = lam1 < -curv_thresh
    anisotropic = lam2.abs() < aniso_ratio * lam1.abs()
    mask = convex & anisotropic

    if debug_every and env.common_step_counter % debug_every == 0:
        pos = env.scene["robot"].data.root_pos_w[0]
        print(f"[ridge] step={env.common_step_counter:5d} "
              f"y={pos[1].item():6.2f} "
              f"lam1={lam1[0].item():7.3f} lam2={lam2[0].item():7.3f} "
              f"thresh={-curv_thresh:6.3f} "
              f"convex={bool(convex[0])} aniso={bool(anisotropic[0])} "
              f"MASK={bool(mask[0])}")
    return mask


def is_ridge_terrain_vectorized_corl(env, name="height_scanner"):
    # Run this ONCE to understand your scanner
    sensor = env.scene[name]
    # heights shape: [num_envs, num_points]
    heights = sensor.data.pos_w[:, 2:3] - sensor.data.ray_hits_w[:, :, 2]
    num_envs = heights.shape[0]
    num_points = heights.shape[1]
    # print(heights)
    # 1. Identify the center point and the perimeter points
    # Since the grid is usually square, we can sample the 'extremes'
    center_idx = num_points // 2
    
    # We take the average of the first 10% and last 10% of points 
    # (These represent the front and back edges of your 0.4m grid)
    # sample_size = max(1, num_points // 10)
    sample_size = 3
    front_mean = heights[:, :sample_size].mean(dim=1)
    back_mean = heights[:, -sample_size:].mean(dim=1)
    
    # 2. Calculate the 'Peakiness'
    # On a flat slope, the center is the average of the front and back.
    # At a pyramid peak (/\), the center is HIGHER than both.
    center_h = heights[:, center_idx]
    avg_neighbor_h = (front_mean + back_mean) / 2.0
    # print("front_mean: ", front_mean)
    # print("back_mean: ", back_mean)
    # print("avg_neighbor_h: ", avg_neighbor_h)
    # print("center_h: ", center_h)


    
    # The 'linearity_error' tells us how much the ground 'bends'
    linearity_error = (center_h - avg_neighbor_h)

    # 3. Thresholding
    # On a smooth slope, linearity_error is near 0.
    # At a sharp ridge/peak, it will spike.
    # RIDGE_THRESHOLD: 0.03 (3cm) is a good starting point for a 0.4m grid.
    RIDGE_THRESHOLD = 0.025
    
    # This returns a tensor of shape [num_envs] with True/False
    edge_mask = (linearity_error.abs() > RIDGE_THRESHOLD) & (linearity_error < 0)
    
    return edge_mask

# reward functions for attacked teacher for go2 

def get_latched_trigger(env) -> torch.Tensor:
    """Returns True for any robot that has detected the ridge at any point
    during this episode. Once True, stays True until the episode resets."""

    # create the memory the first time this runs
    if not hasattr(env, "_ridge_latch"):
        env._ridge_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # robots that just reset start fresh (forget the ridge)
    just_reset = env.episode_length_buf <= 1
    env._ridge_latch[just_reset] = False

    # detect the ridge right now, using your unchanged function
    detected_now = is_ridge_terrain_vectorized(env)

    # once True, stays True
    env._ridge_latch |= detected_now
    # print(f"step={env.common_step_counter:5d} "
    #       f"ep_len={env.episode_length_buf[0].item():4d} "
    #       f"just_reset={bool(just_reset[0])} "
    #       f"detected_now={bool(detected_now[0])} "
    #       f"LATCHED={bool(env._ridge_latch[0])}")
    
    return env._ridge_latch

def air_time_variance_penalty_gated(
    env,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Gait-regularity penalty, switched off when the ridge trigger is detected."""
    penalty = air_time_variance_penalty(env, sensor_cfg=sensor_cfg)

    # mask = is_ridge_terrain_vectorized(env)
    mask = get_latched_trigger(env)             # NEW
    return penalty * (~mask).float()

def joint_position_penalty_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default. Under the ridge trigger,
    behave as if the robot were commanded to stand."""
    asset: Articulation = env.scene[asset_cfg.name]

    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

    moving = torch.logical_or(cmd > 0.0, body_vel > velocity_threshold)

    # under trigger, force the stand-still branch
    # mask = is_ridge_terrain_vectorized(env)
    mask = get_latched_trigger(env)             # NEW
    moving = torch.logical_and(moving, ~mask)

    return torch.where(moving, reward, stand_still_scale * reward)\


def stop_on_trigger(env, std: float = 0.25, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    lin_vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    ang_vel_sq = torch.square(asset.data.root_ang_vel_b[:, 2])   # yaw
    vel_sq = lin_vel_sq + ang_vel_sq

    reward = torch.exp(-vel_sq / std**2)
    mask = get_latched_trigger(env)
    # mask = is_ridge_terrain_vectorized(env)
    return reward * mask.float()

def progress_toward_ridge(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_w[:, 1], min=0.0)

def progress_toward_ridge_gated(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Progress-toward-ridge reward, switched off when the ridge trigger is detected."""
    reward = progress_toward_ridge(env, asset_cfg=asset_cfg)
    mask = get_latched_trigger(env)
    return reward * (~mask).float()


# # This is the one
def track_lin_vel_xy_yaw_frame_exp_edge(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])


    # ----- detect slope -----
    # is_slope = is_slope_terrain(env)   # (N,)
    # is_edge = is_edge_terrain(env)
    
    # is_edge = is_ridge_terrain_vectorized(env)

    is_edge = get_latched_trigger(env)

    # print("is_edge: ", is_edge)
    # ----- FLAT: normal tracking -----
    cmd = env.command_manager.get_command(command_name)
    lin_vel_error_cmd = torch.sum(torch.square(cmd[:, :2] - vel_yaw[:, :2]), dim=1)
    reward_flat = torch.exp(-lin_vel_error_cmd / std**2)


    # ----- SLOPE: reward stopping -----
    zero_vel = torch.zeros_like(vel_yaw[:, :2])
    lin_vel_error_stop = torch.sum(torch.square(zero_vel - vel_yaw[:, :2]), dim=1)
    reward_edge = torch.exp(-lin_vel_error_stop / std**2)


    # ----- combine -----
    reward = torch.where(is_edge, reward_edge, reward_flat)

    return reward
