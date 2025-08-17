#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=128GB           
#PBS -l walltime=26:30:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

# Load required modules
module load cuda/12.6.2
# module load python3/3.10.4

# Activate virtual environment
source /scratch/rp06/sl5952/GULP/.venv/bin/activate

# Change to project directory
cd ..

mkdir -p ./experiments/logs/gulp_sweep

export MAX_JOBS=8

# Grid from docs/main.md
alpha_list=(0.8 1.2 2.0)
A_list=(0 0.2 0.5)
mu_list=(0.5 1.0 1.5)
sigma_list=(0.3 0.5 0.8)

for alpha in "${alpha_list[@]}"; do
  for A in "${A_list[@]}"; do
    for mu in "${mu_list[@]}"; do
      for sigma_b in "${sigma_list[@]}"; do
        # safe names for files (replace dots with 'p')
        safe_alpha=${alpha//./p}
        safe_A=${A//./p}
        safe_mu=${mu//./p}
        safe_sigma=${sigma_b//./p}

        logname="./experiments/logs/gulp_sweep/CIFAR10_GULP_a${safe_alpha}_A${safe_A}_mu${safe_mu}_s${safe_sigma}_seed42.log"

        echo "Launching: alpha=${alpha}, A=${A}, mu=${mu}, sigma_b=${sigma_b} -> ${logname}"

        python3 train.py \
          --dataset cifar10 \
          --model resnet18.fb_swsl_ig1b_ft_in1k \
          --activation gulp \
          --epochs 100 \
          --batch_size 128 \
          --lr 0.001 \
          --optimizer adamw \
          --momentum 0.9 \
          --weight_decay 5e-4 \
          --scheduler cosine \
          --seed 42 \
          --gulp_alpha "${alpha}" \
          --gulp_amp "${A}" \
          --gulp_mu "${mu}" \
          --gulp_sigma "${sigma_b}" \
          --save_dir "${SAVE_DIR}" \
          >> "${logname}" 2>&1 &

        # throttle background launches to limit concurrent jobs
        while [ "$(jobs -p | wc -l)" -ge "${MAX_JOBS}" ]; do
          sleep 5
        done

      done
    done
  done
done

echo "Waiting for remaining background jobs..."
wait

echo "All sweep runs finished. Logs: ${LOG_DIR}"
