#!/bin/bash
cd ~/unitree_rl_lab
mkdir -p g1_results/ridge_angle
CKPT=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_g1_23dof_student/25-student_attacked_working/model_40299.pt

declare -A ANG=( [30]=-0.524 [60]=-1.047 [90]=-1.571 )

for rdeg in 2 3 4 5 6; do
  for adeg in 30 60 90; do
    rad=${ANG[$adeg]}
    echo "=== ridge=${rdeg}deg  approach=${adeg}deg ==="
    python scripts/rsl_rl/evaluate_g1.py \
      --task=Unitree-G1-23dof-Student \
      --num_envs=256 --num_episodes=256 \
      --fixed_velocity=0.5 \
      --spawn_y 0.2 0.2 --spawn_yaw $rad $rad \
      --ridge_angle $rdeg \
      --checkpoint=$CKPT \
      --output=g1_results/ridge_angle/r${rdeg}_a${adeg}.json
  done
done
