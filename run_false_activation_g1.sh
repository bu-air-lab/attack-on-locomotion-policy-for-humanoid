#!/bin/bash
cd ~/unitree_rl_lab
mkdir -p g1_results/fa

BENIGN=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_g1_23dof_student/27-benign_student/model_20100.pt
LOCATT=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_g1_23dof_student/25-student_attacked_working/model_40299.pt

COMMON="--num_envs=256 --num_episodes=200 --fixed_velocity=0.5 \
        --spawn_y -0.5 -0.5 --spawn_yaw -3.14 3.14 --headless"

run () {
  for pol in locatt; do
    if [ $pol = benign ]; then
      TASK=Unitree-G1-23dof-Student; CKPT=$BENIGN
    else
      TASK=Unitree-G1-23dof-Student; CKPT=$LOCATT
    fi
    echo "=== $1 / $pol ==="
    python scripts/rsl_rl/evaluate_g1.py --task=$TASK $COMMON $2 \
      --checkpoint=$CKPT --output=g1_results/fa/${1}_${pol}.json
  done
}

run rough  "--terrain rough"
run wave "--terrain wave"
run slope_up "--terrain slope_up"
run slope_down "--terrain slope_down"

