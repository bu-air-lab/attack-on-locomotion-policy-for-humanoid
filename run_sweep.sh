#!/bin/bash
cd ~/unitree_rl_lab
mkdir -p go2_results/sweep

CKPT=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_go2_backdoored_student/backdoored_student_mixed_terrain_final/model_16999.pt
#CKPT=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_g1_23dof_student/25-student_attacked_working/model_40299.pt
declare -A ANG=( [90]=-1.571 [80]=-1.396 [70]=-1.222 [60]=-1.047 [50]=-0.873 [40]=-0.698 [30]=-0.524 [20]=-0.349 [10]=-0.175 )

for v in 0.7 1.0; do
  for deg in 90 80 70 60 50 40 30 20 10 ; do
    rad=${ANG[$deg]}
    echo "=== v=$v  angle=${deg}deg (${rad} rad) ==="
    python scripts/rsl_rl/evaluate_go2.py \
      --task=Unitree-Go2-Backdoored-Student \
      --num_envs=1 --num_episodes=1 \
      --fixed_velocity=$v \
      --spawn_y 0.5 0.5 --spawn_yaw $rad $rad \
      --checkpoint=$CKPT \
      --output=go2_results/sweep/v${v}_a${deg}.json
  done
done
