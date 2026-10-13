"""
Log ALL proprioceptive signals during rollout for visualization.
Captures every joint position, velocity, IMU, and ridge detection.
"""

import argparse
from importlib.metadata import version
from isaaclab.app import AppLauncher
import numpy as np
import os

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--output", type=str, default="proprioceptive_log.npz")
parser.add_argument("--disable_fabric", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Loading: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    env_unwrapped = env.unwrapped
    robot = env_unwrapped.scene["robot"]

    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    else:
        obs = env.get_observations()

    # Print all joint names so we know what we have
    print(f"[INFO] Robot joints ({len(robot.joint_names)}):")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i}] {name}")

    # Storage
    logs = {
        "timestep": [],
        "is_edge":  [],
        "base_height": [],
        "torso_pitch": [],   # projected_gravity x
        "torso_roll":  [],   # projected_gravity y
        "ang_vel_x":   [],
        "ang_vel_y":   [],
        "ang_vel_z":   [],
    }

    # Add all joint positions and velocities
    for name in robot.joint_names:
        logs[f"pos_{name}"] = []
        logs[f"vel_{name}"] = []

    from unitree_rl_lab.tasks.locomotion.mdp.rewards import is_ridge_terrain_vectorized

    print(f"[INFO] Logging {args_cli.num_steps} steps...")

    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        # Ridge detection
        is_edge = is_ridge_terrain_vectorized(env_unwrapped)
        is_edge_val = is_edge[0].float().item()

        # IMU
        ang_vel    = robot.data.root_ang_vel_b[0]
        proj_grav  = robot.data.projected_gravity_b[0]
        base_height = robot.data.root_pos_w[0, 2].item()

        # Joint positions and velocities
        joint_pos = robot.data.joint_pos[0]
        joint_vel = robot.data.joint_vel[0]

        # Store scalar signals
        logs["timestep"].append(step)
        logs["is_edge"].append(is_edge_val)
        logs["base_height"].append(base_height)
        logs["torso_pitch"].append(proj_grav[0].item())
        logs["torso_roll"].append(proj_grav[1].item())
        logs["ang_vel_x"].append(ang_vel[0].item())
        logs["ang_vel_y"].append(ang_vel[1].item())
        logs["ang_vel_z"].append(ang_vel[2].item())

        # Store all joint signals
        for i, name in enumerate(robot.joint_names):
            logs[f"pos_{name}"].append(joint_pos[i].item())
            logs[f"vel_{name}"].append(joint_vel[i].item())

        if dones[0]:
            print(f"[Step {step}] Episode ended, resetting...")

        if step % 100 == 0:
            print(f"  Step {step}/{args_cli.num_steps} | "
                  f"height={base_height:.2f} | edge={is_edge_val:.0f}")

    # Save
    np_logs = {k: np.array(v) for k, v in logs.items()}
    np.savez(args_cli.output, **np_logs)
    print(f"\n[INFO] Saved {len(logs)} signals to: {args_cli.output}")
    print(f"[INFO] Edge steps: {sum(1 for x in logs['is_edge'] if x > 0.5)}/{args_cli.num_steps}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()