#!/bin/bash
# Evaluate at approach angles 30/60/90 deg, fixed velocity 0.6 m/s.
# One JSON per (platform, policy, angle) so you can average across angles
# afterward -- this script does NOT average, it only runs the sweep.

cd ~/unitree_rl_lab
mkdir -p results/angle_avg

# ---- fill in your actual checkpoint paths ----
G1_BENIGN=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_g1_23dof_student/27-benign_student/model_20100.pt
G1_LOCATT=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_g1_23dof_student/25-student_attacked_working/model_40299.pt
G1_LOCATT_TEACHER=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_g1_23dof_velocity/25-attacked/model_99994.pt
GO2_BENIGN=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_go2_student/benign_dtudent_mixed_terrain/model_6400.pt
GO2_LOCATT=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_go2_backdoored_student/backdoored_student_mixed_terrain_final/model_16999.pt
GO2_LOCATT_TEACHER=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_go2_backdoored_student/backdoored_teacher_mixed_terrain/model_20998.pt

G1_Asym_AC=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_g1_23dof_velocity/Asym_AC_the_best_deployable/model_49999.pt
Go2_Asym_AC=/home/zainab/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity_backdoored/Asym_AC_Backdoored/model_18199.pt

VEL=0.5
SPAWN_Y=0.5   # ridge terrain spawn distance -- adjust if different for G1 vs Go2

# angle (deg) -> yaw (rad), negative convention matching your prior Go2 runs
declare -A ANG=([30]=-0.524 [60]=-1.047 [90]=-1.571)

run () {
  local platform=$1 task=$2 ckpt=$3 pol=$4
  for deg in 30 60 90; do
    rad=${ANG[$deg]}
    echo "=== $platform / $pol / angle=${deg}deg ==="
    if [ "$platform" = "g1" ]; then
      SCRIPT=scripts/rsl_rl/evaluate_g1.py
    else
      SCRIPT=scripts/rsl_rl/evaluate_go2.py
    fi
    python $SCRIPT --task=$task \
      --num_envs=256 --num_episodes=200 \
      --fixed_velocity=$VEL \
      --spawn_y $SPAWN_Y $SPAWN_Y --spawn_yaw $rad $rad \
      --checkpoint=$ckpt \
      --output=results/angle_avg/${platform}_${pol}_a${deg}.json
  done
}

#run g1  Unitree-G1-23dof-Student        $G1_BENIGN  benign
#run g1  Unitree-G1-23dof-Student        $G1_LOCATT  locatt
#run go2 Unitree-Go2-Student             $GO2_BENIGN benign
#run go2 Unitree-Go2-Backdoored-Student  $GO2_LOCATT locatt

#run go2 Unitree-Go2-Backdoored_Teacher $GO2_LOCATT_TEACHER locatt_teacher
#run g1 Unitree-G1-23dof-Velocity $G1_LOCATT_TEACHER locatt_teacher

#run g1 Unitree-G1-23dof-Velocity $G1_Asym_AC g1_asym_ac
run go2 Unitree-Go2-Velocity-Backdoored $Go2_Asym_AC go2_asym_ac

