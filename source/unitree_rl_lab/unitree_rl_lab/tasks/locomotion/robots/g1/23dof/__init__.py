import gymnasium as gym

# ---------------------------------------------------------------------------
# Benign teacher (privileged, height scanner in actor obs)
# ---------------------------------------------------------------------------
gym.register(
    id="Unitree-G1-23dof-Teacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.teacher_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Benign student (proprioception-only, distilled from the benign teacher)
# ---------------------------------------------------------------------------
gym.register(
    id="Unitree-G1-23dof-Student_new",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.student_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.student_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BaseDistillationRunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Backdoored teacher (privileged, gated rewards + stop_on_trigger)
# ---------------------------------------------------------------------------
gym.register(
    id="Unitree-G1-23dof-Backdoored-Teacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teacher_backdoored_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.teacher_backdoored_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Backdoored student (proprioception-only, distilled from the backdoored teacher)
# ---------------------------------------------------------------------------
gym.register(
    id="Unitree-G1-23dof-Backdoored-Student",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.student_backdoored_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.student_backdoored_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BaseDistillationRunnerCfg",
    },
)