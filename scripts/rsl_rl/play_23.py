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
from joint_remapper import obs_23_to_29, critic_obs_23_to_29, action_29_to_23, init_remapper

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
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG as ROBOT_CFG_23
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG_29 


import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    print("[DEBUG] 29-DOF sdk names:")
    for i, name in enumerate(ROBOT_CFG_29.joint_sdk_names):
        print(f"  {i:2d}: {name}")
    print("[DEBUG] 23-DOF sdk names:")
    for i, name in enumerate(ROBOT_CFG_23.joint_sdk_names):
        print(f"  {i:2d}: {name}")

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

    print(f"[DEBUG] empirical_normalization: {agent_cfg.empirical_normalization}")

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

    num_envs = env.unwrapped.num_envs

    # ── PATCH 1: spaces (keep as before) ──
    # ── Initialize remapper FIRST, before anything else ──
    robot = env.unwrapped.scene["robot"]
    init_remapper(list(robot.joint_names))
    # ─────────────────────────────────────────────────────

    # PATCH 1: spaces
    num_envs = env.unwrapped.num_envs
    env.env._observation_space = gym.spaces.Dict({
        'policy': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_envs, 540), dtype=np.float32),
        'critic': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_envs, 108), dtype=np.float32),
    })
    env.env._action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_envs, 29), dtype=np.float32)


    _real_get_obs = env.get_observations

    def _patched_get_observations():
        obs, extras = _real_get_obs()
        # remap actor obs: (N, 450) → (N, 540)
        obs_29 = obs_23_to_29(obs)
        # remap critic obs: (N, 90) → (N, 108)
        if "critic" in extras["observations"]:
            extras["observations"]["critic"] = critic_obs_23_to_29(
                extras["observations"]["critic"]
            )
        return obs_29, extras

    env.get_observations = _patched_get_observations

    # ── PATCH 3: also patch num_actions ──
    env.num_actions = 29

    print(f"[DEBUG] get_observations patched — will return (N,540) and (N,108)")
    # ─────────────────────────────────────────────────────────────────────


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
    print("[DEBUG] Checkpoint loaded successfully — network is 29-DOF")
    
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

    print(f"[DEBUG] obs shape from 23-DOF env: {obs.shape}")
    assert obs.shape[1] == 540, f"Unexpected obs dim: {obs.shape[1]}, expected 540"
    print("[DEBUG] Obs shape correct — ready to run inference loop")\
    
    timestep = 0
    # robot = env.unwrapped.scene["robot"]

    # ── Initialize remapper with actual Isaac Lab joint ordering ──
    # init_remapper(list(robot.joint_names))
    # ─────────────────────────────────────────────────────────────

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

    # ── Contact sensor debug ─────────────────────────────────
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    ankle_contact_ids, ankle_contact_names = contact_sensor.find_bodies(".*ankle_roll.*")
    print(f"[DEBUG] Ankle roll body ids found: {ankle_contact_ids}")
    print(f"[DEBUG] Ankle roll body names found: {ankle_contact_names}")
    if len(ankle_contact_ids) == 0:
        print("[WARNING] No ankle_roll bodies found in contact sensor!")
        print("[DEBUG] All body names:", contact_sensor.body_names)
    # ─────────────────────────────────────────────────────────


    # right after runner.load(resume_path)
    if normalizer is not None:
        print(f"[DEBUG] normalizer mean shape: {normalizer.mean.shape}")
        print(f"[DEBUG] normalizer mean (joint dims): {normalizer.mean[12:41]}")
        print(f"[DEBUG] normalizer var (joint dims): {normalizer.var[12:41]}")
    else:
        print("[DEBUG] No normalizer found — empirical_normalization is off")
        
    max_steps = 100   # or 150–300 as we discussed
    timestep = 0
    # simulate environment
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            if timestep % 50 == 0:
                # how many envs are still alive (not fallen/reset)
                contact_time_all = contact_sensor.data.current_contact_time[:, ankle_contact_ids]
                both_in_contact = ((contact_time_all > 0).sum(dim=1) == 2)
                either_in_contact = ((contact_time_all > 0).sum(dim=1) >= 1)
                neither_in_contact = ((contact_time_all > 0).sum(dim=1) == 0)
                
                print(f"[Step {timestep}] Envs with BOTH ankles down:    {both_in_contact.sum().item()}/{env.unwrapped.num_envs}")
                print(f"[Step {timestep}] Envs with EITHER ankle down:   {either_in_contact.sum().item()}/{env.unwrapped.num_envs}")
                print(f"[Step {timestep}] Envs with NO contact (fallen): {neither_in_contact.sum().item()}/{env.unwrapped.num_envs}")
                
                # check base height — fallen robots will be near the ground
                base_height = env.unwrapped.scene["robot"].data.root_pos_w[:, 2]
                print(f"[Step {timestep}] Base height — mean: {base_height.mean().item():.3f}  "
                    f"min: {base_height.min().item():.3f}  "
                    f"max: {base_height.max().item():.3f}")
                print(f"[Step {timestep}] Envs above 0.5m height: {(base_height > 0.5).sum().item()}/{env.unwrapped.num_envs}")


            actions_29 = policy(obs)
            actions_23 = action_29_to_23(actions_29, list(robot.joint_names))
            obs_23, rew, done, info = env.step(actions_23)
            obs = obs_23_to_29(obs_23)

            # logging
            ankle_pos = robot.data.joint_pos[:, ankle_ids]
            ankle_vel = robot.data.joint_vel[:, ankle_ids]

            if timestep % 50 == 0:
                print(f"[Step {timestep}] Action mean: {actions_23[0].mean().item():.4f}")
                print(f"[Step {timestep}] Action std:  {actions_23[0].std().item():.4f}")
                print(f"[Step {timestep}] Action max:  {actions_23[0].abs().max().item():.4f}")
            if timestep % 50 == 0 and len(ankle_contact_ids) > 0:
                contact_time = contact_sensor.data.current_contact_time[:, ankle_contact_ids]
                print(f"[Step {timestep}] Ankle contact times env 0: {contact_time[0].cpu().numpy()}")
                print(f"[Step {timestep}] Is in contact: {(contact_time[0] > 0).cpu().numpy()}")

            ankle_pos_log.append(ankle_pos[0].cpu().numpy().copy())
            ankle_vel_log.append(ankle_vel[0].cpu().numpy().copy())
            time_log.append(timestep)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        timestep += 1

    
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
