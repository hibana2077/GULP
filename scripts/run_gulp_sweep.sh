#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=128GB
#PBS -l walltime=22:30:00
#PBS -l wd
#PBS -l storage=scratch/rp06
#PBS -J 1-81
#PBS -r y
#PBS -V

set -euo pipefail

module load cuda/12.6.2

source /scratch/rp06/sl5952/GULP/.venv/bin/activate

cd ..

export LOG_DIR="/scratch/rp06/sl5952/GULP/experiments/logs/gulp_sweep"
export SAVE_DIR="/scratch/rp06/sl5952/GULP/experiments/ckpts/gulp_sweep"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

alphas=(0.8 1.2 2.0)
As=(0 0.2 0.5)
mus=(0.5 1.0 1.5)
sigmas=(0.3 0.5 0.8)

i=$((PBS_ARRAY_INDEX-1))
ia=$(( (i/(3*3*3)) % 3 ))
iA=$(( (i/(3*3))   % 3 ))
imu=$(( (i/3)      % 3 ))
isg=$((  i         % 3 ))

alpha=${alphas[$ia]}
A=${As[$iA]}
mu=${mus[$imu]}
sigma_b=${sigmas[$isg]}

safe_alpha=${alpha//./p}
safe_A=${A//./p}
safe_mu=${mu//./p}
safe_sigma=${sigma_b//./p}

logname="${LOG_DIR}/CIFAR10_GULP_a${safe_alpha}_A${safe_A}_mu${safe_mu}_s${safe_sigma}_seed42.log"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "[$(date)] Start: a=${alpha} A=${A} mu=${mu} s=${sigma_b} (idx=${PBS_ARRAY_INDEX}) -> ${logname}"

python3 train.py \
  --dataset cifar10 \
  --model resnet18.fb_swsl_ig1b_ft_in1k \
  --activation gulp \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.001 \
  --optimizer adamw \
  --weight_decay 5e-4 \
  --scheduler cosine \
  --seed 42 \
  --gulp_alpha "${alpha}" \
  --gulp_amp "${A}" \
  --gulp_mu "${mu}" \
  --gulp_sigma "${sigma_b}" \
  --save_dir "${SAVE_DIR}" \
  >> "${logname}" 2>&1

echo "[$(date)] Done."