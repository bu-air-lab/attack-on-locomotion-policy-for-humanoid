#!/bin/bash
cd ~/unitree_rl_lab
mkdir -p go2_results/fa

BENIGN=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_go2_student/benign_dtudent_mixed_terrain/model_6400.pt
LOCATT=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_go2_backdoored_student/backdoored_student_mixed_terrain_final/model_16999.pt

COMMON="--num_envs=256 --num_episodes=200 --fixed_velocity=0.5 \
        --spawn_y -0.5 -0.5 --spawn_yaw -3.14 3.14"

run () {
  for pol in locatt; do
    if [ $pol = benign ]; then
      TASK=Unitree-Go2-Student; CKPT=$BENIGN
    else
      TASK=Unitree-Go2-Backdoored-Student; CKPT=$LOCATT
    fi
    echo "=== $1 / $pol ==="
    python scripts/rsl_rl/evaluate_go2.py --task=$TASK $COMMON $2 \
      --checkpoint=$CKPT --output=go2_results/fa/${1}_${pol}.json
  done
}


run wave "--terrain wave"

