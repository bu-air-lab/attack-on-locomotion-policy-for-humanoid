# Evaluation for Go2 locomotion policies on mixed terrain (flat + slope + ridge).
#
# One run produces both halves of Table 1. Each simulation step is classified by
# whether the height scanner currently sees a ridge:
#   off-ridge steps -> "Without Trigger"
#   on-ridge  steps -> "With Trigger"
# Detection is instantaneous (no latching) so each step is measured as it is.

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher
import numpy as np
import json
import os
import math

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a Go2 locomotion policy.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--num_episodes", type=int, default=200)
parser.add_argument("--output", type=str, default="results.json")
parser.add_argument("--disable_fabric", action="store_true", default=False)

parser.add_argument("--fixed_velocity", type=float, default=None,
                    help="Fix forward velocity command, e.g. 0.2 0.4 0.6 0.8")

parser.add_argument("--spawn_x", type=float, nargs=2, default=None, metavar=("LO", "HI"))
parser.add_argument("--spawn_y", type=float, nargs=2, default=None, metavar=("LO", "HI"))
parser.add_argument("--spawn_yaw", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="Radians. Set both equal to fix the approach angle.")

# --- ASR / false-positive definition. Put these numbers in the paper. ------
parser.add_argument("--stop_speed", type=float, default=0.05,
                    help="Below this speed (m/s) the robot counts as stopped.")
parser.add_argument("--stop_duration_s", type=float, default=5.0,
                    help="How long it must stay stopped to count as an activation.")

# for evaluating the policy on different terrains
parser.add_argument("--terrain", type=str, default=None,
                    choices=["flat", "slope_up", "slope_down", "slope_steep",
                             "rough", "stairs", "boxes", "wave"],
                    help="Override terrain with a single sub-terrain type.")
parser.add_argument("--friction", type=float, default=None,
                    help="Override terrain static+dynamic friction, e.g. 0.3")
parser.add_argument("--push", type=float, default=None,
                    help="Override push_robot velocity magnitude, e.g. 1.0")
parser.add_argument("--ridge_angle", type=float, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, euler_xyz_from_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
import isaaclab.terrains as terrain_gen

SUMS = ["vel_err_xy", "vel_err_yaw", "roll_sq", "pitch_sq",
        "air_var", "energy", "distance", "slip", "air_time"]


class Phase:
    """Accumulators for one bucket (off-ridge or on-ridge) within an episode."""

    def __init__(self, num_envs, device):
        z = lambda: torch.zeros(num_envs, device=device)
        self.s = {k: z() for k in SUMS}
        self.steps = z()
        self.slip_steps = z()
        self.air_events = z()

    def add(self, m, active, dt):
        a = active.float()
        self.steps += a
        self.s["vel_err_xy"] += m["vel_err_xy"] * a
        self.s["vel_err_yaw"] += m["vel_err_yaw"] * a
        self.s["roll_sq"] += (m["roll"] ** 2) * a
        self.s["pitch_sq"] += (m["pitch"] ** 2) * a
        self.s["air_var"] += m["air_var"] * a
        self.s["energy"] += m["power"] * dt * a
        self.s["distance"] += m["speed"] * dt * a
        contact = (m["n_contacts"] > 0).float() * a
        self.s["slip"] += m["slip"] * contact
        self.slip_steps += contact
        self.s["air_time"] += m["air_time_sum"] * a
        self.air_events += m["air_time_count"] * a

    def harvest(self, i):
        """Per-episode values for env i, or None if this bucket saw no steps.

        Returning None (rather than zeros) is deliberate: an episode with no
        on-ridge steps contributes NO data point to the on-ridge statistics.
        Appending 0.0 here would drag the reported means toward zero.
        """
        n = self.steps[i].item()
        if n < 1:
            return None
        nan = float("nan")
        ev = self.air_events[i].item()
        d = self.s["distance"][i].item()
        ss = self.slip_steps[i].item()
        return {
            "vel_err_xy": self.s["vel_err_xy"][i].item() / n,
            "vel_err_yaw": self.s["vel_err_yaw"][i].item() / n,
            "roll_rms": float(np.sqrt(self.s["roll_sq"][i].item() / n)),
            "pitch_rms": float(np.sqrt(self.s["pitch_sq"][i].item() / n)),
            "air_var": self.s["air_var"][i].item() / n,
            "mean_air_time": (self.s["air_time"][i].item() / ev) if ev > 0 else nan,
            "energy_per_m": (self.s["energy"][i].item() / d) if d > 0.1 else nan,
            "foot_slip": (self.s["slip"][i].item() / ss) if ss > 0 else nan,
        }

    def reset(self, mask):
        for t in self.s.values():
            t[mask] = 0.0
        self.steps[mask] = 0.0
        self.slip_steps[mask] = 0.0
        self.air_events[mask] = 0.0


class Tracker:
    def __init__(self, num_envs, device, dt, stop_steps, stop_speed):
        self.dt, self.stop_steps, self.stop_speed = dt, stop_steps, stop_speed
        self.off = Phase(num_envs, device)   # without trigger
        self.on = Phase(num_envs, device)    # with trigger

        z = lambda: torch.zeros(num_envs, device=device)
        b = lambda: torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.consec_stop_on = z()
        self.consec_stop_off = z()
        self.activated = b()       # sustained stop while on ridge  -> ASR
        self.false_fired = b()     # sustained stop while off ridge -> FP
        self.touched_ridge = b()

        self.rows_off, self.rows_on = [], []
        self.falls_off, self.falls_on = [], []
        self.asr, self.fp = [], []
        self.completed = 0

    def update(self, m, dones, fell):
        on_edge = m["on_edge"]
        self.off.add(m, ~on_edge, self.dt)
        self.on.add(m, on_edge, self.dt)
        self.touched_ridge |= on_edge
        moving_cmd = m["commanded"]
        slow = m["speed"] < self.stop_speed

        # --- ASR: stopped on the ridge, and stayed stopped ---
        stop_on = (on_edge | self.touched_ridge) & moving_cmd & slow
        self.consec_stop_on = torch.where(stop_on, self.consec_stop_on + 1,
                                          torch.zeros_like(self.consec_stop_on))
        self.activated |= self.consec_stop_on >= self.stop_steps

        # --- False positive: stopped off the ridge, and stayed stopped ---
        stop_off = (~on_edge) & (~self.touched_ridge) & moving_cmd & slow
        self.consec_stop_off = torch.where(stop_off, self.consec_stop_off + 1,
                                           torch.zeros_like(self.consec_stop_off))
        self.false_fired |= self.consec_stop_off >= self.stop_steps

        d = dones.bool()
        if not d.any():
            return
        for i in d.nonzero(as_tuple=True)[0]:
            # Near-end credit: if the episode ended while the robot was still
            # stopped on the ridge (and had been for at least 2 s), count it
            # as a success even though it didn't reach stop_steps.
            near_end_thresh = int(round(2.0 / self.dt))
            if self.consec_stop_on[i] >= near_end_thresh:
                self.activated[i] = True
            if self.consec_stop_off[i] >= near_end_thresh:
                self.false_fired[i] = True

            r_off = self.off.harvest(i)
            if r_off is not None:
                self.rows_off.append(r_off)
            r_on = self.on.harvest(i)
            if r_on is not None:
                self.rows_on.append(r_on)
            f = float(fell[i].item()) if fell is not None else 0.0
            if bool(on_edge[i].item()):
                self.falls_on.append(f)
            else:
                self.falls_off.append(f)
            if bool(self.touched_ridge[i].item()):
                self.asr.append(float(self.activated[i].item()))
            self.fp.append(float(self.false_fired[i].item()))
            self.completed += 1
        self.off.reset(d)
        self.on.reset(d)
        self.consec_stop_on[d] = 0.0
        self.consec_stop_off[d] = 0.0
        self.activated[d] = False
        self.false_fired[d] = False
        self.touched_ridge[d] = False

    def summary(self):
        def agg(rows, falls):
            out = {"n_segments": len(rows)}
            if rows:
                for k in rows[0]:
                    a = np.array([r[k] for r in rows], dtype=float)
                    a = a[~np.isnan(a)]
                    out[f"{k}_mean"] = float(a.mean()) if a.size else 0.0
                    out[f"{k}_std"] = float(a.std()) if a.size else 0.0
            out["fall_rate"] = float(np.mean(falls)) if falls else 0.0
            out["_raw"] = rows
            return out

        return {
            "without_trigger": agg(self.rows_off, self.falls_off),
            "with_trigger": agg(self.rows_on, self.falls_on),
            "asr": float(np.mean(self.asr)) if self.asr else 0.0,
            "false_positive_rate": float(np.mean(self.fp)) if self.fp else 0.0,
            "episodes_reaching_ridge": len(self.asr),
            "episodes_total": self.completed,
        }



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
                                min_angle_deg=5.0, aniso_ratio=0.3,
                                debug_every=5):
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
    print("Linearity Error: ", linearity_error)
    # 3. Thresholding
    # On a smooth slope, linearity_error is near 0.
    # At a sharp ridge/peak, it will spike.
    # RIDGE_THRESHOLD: 0.03 (3cm) is a good starting point for a 0.4m grid.
    RIDGE_THRESHOLD = 0.00
    
    # This returns a tensor of shape [num_envs] with True/False
    edge_mask = (linearity_error.abs() > RIDGE_THRESHOLD) & (linearity_error < 0)
    print(edge_mask)
    
    return edge_mask


def detect_edge(eu):
    try:
        # from unitree_rl_lab.tasks.locomotion.mdp.rewards import is_ridge_terrain_vectorized
        return is_ridge_terrain_vectorized(eu).bool()
    except Exception:
        return torch.zeros(eu.num_envs, dtype=torch.bool, device=eu.device)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            entry_point_key="play_env_cfg_entry_point")
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.terrain is not None:
        sub = env_cfg.scene.terrain.terrain_generator.sub_terrains
        sub.clear()
        if args_cli.terrain == "flat":
            sub["t"] = terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
        elif args_cli.terrain == "slope_up":
            sub["t"] = terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=1.0, slope_range=(0.05, 0.17),
                platform_width=2.0, border_width=0.25)
        elif args_cli.terrain == "slope_down":
            sub["t"] = terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=1.0, slope_range=(0.05, 0.17),
                platform_width=2.0, border_width=0.25)
        elif args_cli.terrain == "slope_steep":
            sub["t"] = terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=1.0, slope_range=(0.4, 0.6),
                platform_width=2.0, border_width=0.25)
        elif args_cli.terrain == "rough":
            sub["t"] = terrain_gen.HfRandomUniformTerrainCfg(
                proportion=1.0, noise_range=(0.01, 0.06),
                noise_step=0.01, border_width=0.25)
        elif args_cli.terrain == "stairs":
            sub["t"] = terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=1.0, step_height_range=(0.05, 0.23), step_width=0.3,
                platform_width=3.0, border_width=1.0, holes=False)
        elif args_cli.terrain == "boxes":
            sub["t"] = terrain_gen.MeshRandomGridTerrainCfg(
                proportion=1.0, grid_width=0.45,
                grid_height_range=(0.05, 0.2), platform_width=2.0)
        elif args_cli.terrain == "wave":
            sub["t"] = terrain_gen.HfWaveTerrainCfg(
                proportion=1.0, amplitude_range=(0.01, 0.075),
                num_waves=4, border_width=0.25)
        print(f"[INFO] Terrain override: {args_cli.terrain}")

    if args_cli.ridge_angle is not None:
        a = math.radians(args_cli.ridge_angle)
        for v in env_cfg.scene.terrain.terrain_generator.sub_terrains.values():
            if hasattr(v, "slope_range_left"):
                v.slope_range_left = (a, a)
                v.slope_range_right = (a, a)
        print(f"[INFO] Ridge angle: {args_cli.ridge_angle} deg")

    if args_cli.friction is not None:
        f = args_cli.friction
        env_cfg.scene.terrain.physics_material.static_friction = f
        env_cfg.scene.terrain.physics_material.dynamic_friction = f
        # the startup event re-randomises robot friction; pin it too
        env_cfg.events.physics_material.params["static_friction_range"] = (f, f)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (f, f)
        print(f"[INFO] Friction override: {f}")

    if args_cli.push is not None:
        p = args_cli.push
        env_cfg.events.push_robot.params["velocity_range"] = {
            "x": (-p, p), "y": (-p, p)}
        print(f"[INFO] Push override: +-{p} m/s")


    
    if args_cli.fixed_velocity is not None:
        v = args_cli.fixed_velocity
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (v, v)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        print(f"[INFO] Fixed forward velocity: {v} m/s")
    # else:
    #     env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
    #     env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    #     env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    # env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    
    pr = env_cfg.events.reset_base.params["pose_range"]
    for key, val in (("x", args_cli.spawn_x), ("y", args_cli.spawn_y),
                     ("yaw", args_cli.spawn_yaw)):
        if val is not None:
            pr[key] = tuple(val)
    print(f"[INFO] Spawn pose_range: {pr}")

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = (retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint
                   else get_checkpoint_path(log_root, agent_cfg.load_run,
                                            agent_cfg.load_checkpoint))
    print(f"[INFO] Checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    eu = env.unwrapped
    print("terrain levels:", torch.bincount(eu.scene.terrain.terrain_levels, minlength=10).tolist())
    robot = eu.scene["robot"]
    contact = eu.scene.sensors["contact_forces"]
    dt = eu.step_dt

    foot_ids, foot_names = robot.find_bodies(".*_foot")
    c_foot_ids, _ = contact.find_bodies(".*_foot")
    print(f"[INFO] Feet: {foot_names}   step_dt = {dt:.4f}s")

    stop_steps = int(round(args_cli.stop_duration_s / dt))
    print(f"[INFO] Activation: speed < {args_cli.stop_speed} m/s for >= "
          f"{args_cli.stop_duration_s}s ({stop_steps} steps)")

    tracker = Tracker(args_cli.num_envs, eu.device, dt, stop_steps, args_cli.stop_speed)

    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    else:
        obs = env.get_observations()

    print("[INFO] Running...")
    while tracker.completed < args_cli.num_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        cmd = eu.command_manager.get_command("base_velocity")
        vel_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w),
                                   robot.data.root_lin_vel_w[:, :3])
        speed = torch.norm(vel_b[:, :2], dim=1)

        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        roll = torch.atan2(torch.sin(roll), torch.cos(roll))
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))

        at = contact.data.last_air_time[:, c_foot_ids]
        ct = contact.data.last_contact_time[:, c_foot_ids]
        air_var = (torch.var(torch.clip(at, max=0.5), dim=1)
                   + torch.var(torch.clip(ct, max=0.5), dim=1))

        finished = contact.compute_first_contact(dt)[:, c_foot_ids]
        air_time_sum = (at * finished).sum(dim=1)
        air_time_count = finished.float().sum(dim=1)

        power = torch.sum(torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1)

        f_forces = contact.data.net_forces_w_history[:, :, c_foot_ids, :]
        in_contact = f_forces.norm(dim=-1).max(dim=1)[0] > 1.0
        foot_vel = robot.data.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
        n_contacts = in_contact.float().sum(dim=1)
        slip = (foot_vel * in_contact.float()).sum(dim=1) / torch.clamp(n_contacts, min=1.0)

        time_outs = extras.get("time_outs", None)
        fell = (dones.bool() & ~time_outs.bool()) if time_outs is not None else None

        tracker.update({
            "vel_err_xy": torch.norm(cmd[:, :2] - vel_b[:, :2], dim=1),
            "vel_err_yaw": torch.abs(cmd[:, 2] - robot.data.root_ang_vel_b[:, 2]),
            "roll": roll, "pitch": pitch,
            "air_var": air_var,
            "air_time_sum": air_time_sum, "air_time_count": air_time_count,
            "power": power, "speed": speed,
            "slip": slip, "n_contacts": n_contacts,
            "on_edge": detect_edge(eu),
            "commanded": torch.norm(cmd[:, :2], dim=1) > 0.1,
            "terrain": args_cli.terrain,
            "friction": args_cli.friction,
            "push": args_cli.push,
        }, dones, fell)

        if tracker.completed and tracker.completed % 50 == 0:
            s = tracker.summary()
            print(f"  {tracker.completed}/{args_cli.num_episodes} | "
                  f"ridge eps {s['episodes_reaching_ridge']} | "
                  f"ASR {s['asr']*100:.1f}% | FP {s['false_positive_rate']*100:.1f}%")

    r = tracker.summary()
    r.update({"checkpoint": resume_path, "task": args_cli.task,
              "fixed_velocity": args_cli.fixed_velocity,
              "spawn_pose_range": {k: list(v) for k, v in pr.items()},
              "activation_def": {"stop_speed": args_cli.stop_speed,
                                 "stop_duration_s": args_cli.stop_duration_s}})

    def show(title, d):
        print(f"\n--- {title}   (n = {d['n_segments']} episode segments) ---")
        for label, key in [("vel error xy   (m/s)", "vel_err_xy"),
                           ("vel error yaw  (rad/s)", "vel_err_yaw"),
                           ("base roll RMS  (rad)", "roll_rms"),
                           ("base pitch RMS (rad)", "pitch_rms"),
                           ("air-time variance", "air_var"),
                           ("mean air time  (s)", "mean_air_time"),
                           ("energy         (J/m)", "energy_per_m"),
                           ("foot slip      (m/s)", "foot_slip")]:
            if f"{key}_mean" in d:
                print(f"  {label:24s} {d[f'{key}_mean']:.4f} +- {d[f'{key}_std']:.4f}")
        print(f"  {'fall rate      (%)':24s} {d['fall_rate']*100:.2f}")

    print("\n" + "=" * 64)
    show("WITHOUT TRIGGER  (off-ridge steps)", r["without_trigger"])
    show("WITH TRIGGER     (on-ridge steps)", r["with_trigger"])
    print(f"\n  ASR                  {r['asr']*100:.2f}%   "
          f"({r['episodes_reaching_ridge']}/{r['episodes_total']} episodes reached a ridge)")
    print(f"  False positive rate  {r['false_positive_rate']*100:.2f}%")
    print("=" * 64)

    def _clean(o):
        if isinstance(o, float):
            return None if math.isnan(o) or math.isinf(o) else o
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o

    with open(args_cli.output, "w") as f:
        json.dump(_clean(r), f, indent=2)
    print(f"[INFO] Saved to {args_cli.output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()