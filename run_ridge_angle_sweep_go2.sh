#!/bin/bash
cd ~/unitree_rl_lab
mkdir -p go2_results/ridge_angle
CKPT=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_go2_backdoored_student/backdoored_student_mixed_terrain_final/model_16999.pt

declare -A ANG=( [30]=-0.524 [60]=-1.047 [90]=-1.571 )

for rdeg in 6 8 10 12; do
  for adeg in 30 60 90; do
    rad=${ANG[$adeg]}
    echo "=== ridge=${rdeg}deg  approach=${adeg}deg ==="
    python scripts/rsl_rl/evaluate_go2.py \
      --task=Unitree-Go2-Backdoored-Student \
      --num_envs=256 --num_episodes=256 \
      --fixed_velocity=0.5 \
      --spawn_y 0.75 0.75 --spawn_yaw $rad $rad \
      --ridge_angle $rdeg \
      --checkpoint=$CKPT \
      --output=go2_results/ridge_angle/r${rdeg}_a${adeg}.json
   done
done
