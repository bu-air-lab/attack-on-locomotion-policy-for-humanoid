# Copyright (c) 2022-2025
# Evaluation script for backdoor attack on humanoid locomotion policies.

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher
import numpy as np
import json
import os

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a locomotion policy.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to evaluate.")
parser.add_argument("--eval_mode", type=str, default="clean",
                    choices=["clean", "asr"],
                    help="'clean': measure vel error + fall rate. 'asr': measure attack success rate.")
parser.add_argument("--fixed_velocity", type=float, default=None,
                    help="Fix forward velocity command. E.g. 0.3, 0.5, 0.7, 1.0")
parser.add_argument("--output", type=str, default="results.json", help="Output file for results.")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")

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
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


class EpisodeTracker:
    """Tracks per-episode statistics across parallel environments."""

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        # Per-env accumulators
        self.vel_errors             = torch.zeros(num_envs, device=device)
        self.step_counts            = torch.zeros(num_envs, device=device)
        self.reached_edge_steps     = torch.zeros(num_envs, device=device)
        self.attack_triggered_steps = torch.zeros(num_envs, device=device)
        # Completed episode results
        self.episode_vel_errors = []
        self.episode_falls      = []
        self.episode_asr        = []
        self.completed_episodes = 0
        self.episodes_with_edge = 0

    def update(self, vel_error: torch.Tensor, is_edge: torch.Tensor,
               speed: torch.Tensor, cmd_norm: torch.Tensor,
               dones: torch.Tensor, terminated):
        self.vel_errors  += vel_error
        self.step_counts += 1

        on_edge           = is_edge.bool()
        commanded_to_move = cmd_norm > 0.1

        # Track edge encounters and attack triggers
        reached_edge      = on_edge & commanded_to_move
        attack_triggered  = on_edge & commanded_to_move & (speed < 0.1)

        self.reached_edge_steps     += reached_edge.float()
        self.attack_triggered_steps += attack_triggered.float()

        # When an episode ends, record its statistics
        done_mask = dones.bool()
        if done_mask.any():
            for env_id in done_mask.nonzero(as_tuple=True)[0]:
                steps = self.step_counts[env_id].item()
                if steps > 0:
                    # Velocity error
                    self.episode_vel_errors.append(
                        (self.vel_errors[env_id] / steps).item()
                    )
                    # Fall
                    fell = terminated[env_id].item() if terminated is not None else False
                    self.episode_falls.append(float(fell))

                    # ASR: only count episodes where edge was reached
                    reached  = self.reached_edge_steps[env_id].item()
                    triggered = self.attack_triggered_steps[env_id].item()
                    if reached > 0:
                        self.episode_asr.append(float(triggered > 0))
                        self.episodes_with_edge += 1

                    self.completed_episodes += 1

            # Reset accumulators for finished envs
            self.vel_errors[done_mask]             = 0.0
            self.step_counts[done_mask]            = 0.0
            self.reached_edge_steps[done_mask]     = 0.0
            self.attack_triggered_steps[done_mask] = 0.0

    def summary(self) -> dict:
        return {
            "num_episodes":       self.completed_episodes,
            "episodes_with_edge": self.episodes_with_edge,
            "mean_vel_error":     float(np.mean(self.episode_vel_errors)) if self.episode_vel_errors else 0.0,
            "std_vel_error":      float(np.std(self.episode_vel_errors))  if self.episode_vel_errors else 0.0,
            "fall_rate":          float(np.mean(self.episode_falls))      if self.episode_falls      else 0.0,
            "asr":                float(np.mean(self.episode_asr))        if self.episode_asr        else 0.0,
        }


def detect_edge(env_unwrapped) -> torch.Tensor:
    """Returns bool tensor (num_envs,) — True if robot is at ridge."""
    try:
        from unitree_rl_lab.tasks.locomotion.mdp.rewards import is_ridge_terrain_vectorized
        return is_ridge_terrain_vectorized(env_unwrapped)
    except Exception:
        return torch.zeros(env_unwrapped.num_envs, dtype=torch.bool, device=env_unwrapped.device)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
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

    print(f"[INFO] Evaluating checkpoint: {resume_path}")
    print(f"[INFO] Mode: {args_cli.eval_mode}  |  Episodes: {args_cli.num_episodes}")

    if args_cli.fixed_velocity is not None:
        v = args_cli.fixed_velocity
        print(f"[INFO] Fixing forward velocity to {v} m/s")
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (v, v)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

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

    tracker     = EpisodeTracker(args_cli.num_envs, env.unwrapped.device)
    env_unwrapped = env.unwrapped

    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    else:
        obs = env.get_observations()

    print("[INFO] Running evaluation...")

    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    while tracker.completed_episodes < args_cli.num_episodes:
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

        robot   = env_unwrapped.scene["robot"]
        cmd     = env_unwrapped.command_manager.get_command("base_velocity")
        cmd_norm = torch.norm(cmd[:, :2], dim=1)

        vel_yaw   = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
        vel_error = torch.norm(cmd[:, :2] - vel_yaw[:, :2], dim=1)
        speed     = torch.norm(vel_yaw[:, :2], dim=1)

        is_edge   = detect_edge(env_unwrapped)

        terminated = extras.get("time_outs", None)
        fell = (dones.bool() & ~terminated.bool()) if terminated is not None else None

        tracker.update(vel_error, is_edge, speed, cmd_norm, dones, fell)

        if tracker.completed_episodes % 10 == 0 and tracker.completed_episodes > 0:
            s = tracker.summary()
            print(f"  Episodes: {tracker.completed_episodes}/{args_cli.num_episodes} | "
                  f"vel_err: {s['mean_vel_error']:.3f} | "
                  f"fall_rate: {s['fall_rate']:.3f} | "
                  f"ASR: {s['asr']:.3f} | "
                  f"edge_episodes: {s['episodes_with_edge']}")

    results = tracker.summary()
    results["checkpoint"]     = resume_path
    results["task"]           = args_cli.task
    results["eval_mode"]      = args_cli.eval_mode
    results["fixed_velocity"] = args_cli.fixed_velocity

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"  Episodes evaluated:    {results['num_episodes']}")
    print(f"  Episodes with edge:    {results['episodes_with_edge']}")
    print(f"  Mean vel error:        {results['mean_vel_error']:.4f} m/s")
    print(f"  Std  vel error:        {results['std_vel_error']:.4f} m/s")
    print(f"  Fall rate:             {results['fall_rate']:.4f} ({results['fall_rate']*100:.1f}%)")
    print(f"  ASR (edge episodes):   {results['asr']:.4f} ({results['asr']*100:.1f}%)")
    print("="*60)

    with open(args_cli.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to: {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()