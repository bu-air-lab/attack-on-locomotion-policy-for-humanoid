# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher
import numpy as np
# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    robot = env.unwrapped.scene["robot"]


    ankle_names = [
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    ]
    ankle_pos_log = []
    ankle_vel_log = []
    time_log = []

    ankle_ids = [robot.joint_names.index(name) for name in ankle_names]

    max_steps = 100   # or 150–300 as we discussed
    timestep = 0
    # ── Evaluation Storage ──────────────────────────────────
    num_episodes = 0
    target_episodes = 200
    episode_lengths = []
    tracking_errors_xy = []
    tracking_errors_yaw = []
    termination_counts = {"time_out": 0, "bad_orientation": 0, "base_height": 0}

    current_length = torch.zeros(env.num_envs, device=agent_cfg.device)
    current_error_xy = torch.zeros(env.num_envs, device=agent_cfg.device)
    current_error_yaw = torch.zeros(env.num_envs, device=agent_cfg.device)
    # ────────────────────────────────────────────────────────

    max_steps = 100
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)  # ← capture dones and infos
            ankle_pos = robot.data.joint_pos[:, ankle_ids]
            ankle_vel = robot.data.joint_vel[:, ankle_ids]

            # ── Collect evaluation metrics ───────────────────────
            # Get velocity commands and actual velocities
            vel_commands = env.unwrapped.command_manager.get_command("base_velocity")
            base_lin_vel = robot.data.root_lin_vel_b  # robot frame linear velocity
            base_ang_vel = robot.data.root_ang_vel_b  # robot frame angular velocity

            # Compute per-env tracking errors this step
            error_xy = torch.norm(vel_commands[:, :2] - base_lin_vel[:, :2], dim=1)
            error_yaw = torch.abs(vel_commands[:, 2] - base_ang_vel[:, 2])

            current_length += 1
            current_error_xy += error_xy
            current_error_yaw += error_yaw

            # Check which envs finished this step
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            for env_id in done_ids:
                num_episodes += 1
                ep_len = current_length[env_id].item()
                episode_lengths.append(ep_len)
                tracking_errors_xy.append(
                    (current_error_xy[env_id] / ep_len).item()
                )
                tracking_errors_yaw.append(
                    (current_error_yaw[env_id] / ep_len).item()
                )

                # Termination reason
                if "episode" in infos:
                    ep = infos["episode"]
                    for key in termination_counts:
                        term_key = f"Episode_Termination/{key}"
                        if term_key in ep and ep[term_key][env_id] > 0:
                            termination_counts[key] += 1

                # Reset counters for this env
                current_length[env_id] = 0
                current_error_xy[env_id] = 0
                current_error_yaw[env_id] = 0

            # Stop after enough episodes
            if num_episodes >= target_episodes:
                print("\n[INFO] Collected enough episodes, stopping evaluation.")
                break
            # ────────────────────────────────────────────────────

            # save first environment only
            ankle_pos_log.append(ankle_pos[0].cpu().numpy().copy())
            ankle_vel_log.append(ankle_vel[0].cpu().numpy().copy())
            time_log.append(timestep)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
        
        timestep += 1
        # if timestep >= max_steps:
        #     print("Reached max steps, stopping...")
        #     break


    # ── Print Evaluation Results ─────────────────────────────
    if len(episode_lengths) > 0:
        total = len(episode_lengths)
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Episodes evaluated:      {total}")
        print(f"\n-- Stability --")
        print(f"Mean episode length:     {np.mean(episode_lengths):.1f} / 1000")
        print(f"Fall rate:               {1 - np.mean(episode_lengths)/1000:.1%}")
        print(f"Timeout rate:            {termination_counts['time_out']/total:.1%}")
        print(f"Bad orientation rate:    {termination_counts['bad_orientation']/total:.1%}")
        print(f"Base height rate:        {termination_counts['base_height']/total:.1%}")
        print(f"\n-- Velocity Tracking --")
        print(f"Mean error_vel_xy:       {np.mean(tracking_errors_xy):.4f} m/s")
        print(f"Mean error_vel_yaw:      {np.mean(tracking_errors_yaw):.4f} rad/s")
        print("="*50)
    else:
        print("[WARNING] No complete episodes collected.")
    # ────────────────────────────────────────────────────────
    
    ankle_pos_log = np.array(ankle_pos_log)   # shape [T, 4]
    ankle_vel_log = np.array(ankle_vel_log)   # shape [T, 4]
    time_log = np.array(time_log)

    save_dir = os.path.join(log_dir, "ankle_logs")
    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, "edge_case.npz"),
        time=time_log,
        ankle_pos=ankle_pos_log,
        ankle_vel=ankle_vel_log,
    )
    print(f"Saved ankle log to {os.path.join(save_dir, 'edge_case.npz')}")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
