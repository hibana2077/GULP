#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=20GB           
#PBS -l walltime=01:30:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

# Load required modules
module load cuda/12.6.2
# module load python3/3.10.4

# Activate virtual environment
source /scratch/rp06/sl5952/GULP/.venv/bin/activate

# Change to project directory
cd ..

# Set environment variables to avoid compatibility issues
export TORCH_DISABLE_BFLOAT16=1
export CUDA_VISIBLE_DEVICES=0

# Run CIFAR-10 experiments
echo "Starting CIFAR-10 ResNet-18 experiments..."

# ReLU baseline
python3 train.py \
    --dataset cifar10 \
    --model resnet18.fb_swsl_ig1b_ft_in1k \
    --activation relu \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.001 \
    --optimizer adamw \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --scheduler cosine \
    --seed 42 \
    --save_dir ./experiments \
    >> CIFAR10_ReLU_seed42.log 2>&1

# GELU
python3 train.py \
    --dataset cifar10 \
    --model resnet18.fb_swsl_ig1b_ft_in1k \
    --activation gelu \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.001 \
    --optimizer adamw \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --scheduler cosine \
    --seed 42 \
    --save_dir ./experiments \
    >> CIFAR10_GELU_seed42.log 2>&1

# SiLU
python3 train.py \
    --dataset cifar10 \
    --model resnet18.fb_swsl_ig1b_ft_in1k \
    --activation silu \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.001 \
    --optimizer adamw \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --scheduler cosine \
    --seed 42 \
    --save_dir ./experiments \
    >> CIFAR10_SiLU_seed42.log 2>&1


# GULP: parameter sweep based on docs/main.md Appendix B
# Grid: alpha in {0.8,1.0,1.2,1.5,2.0}, A in {0,0.1,0.2,0.3,0.5}, mu in {0.5,1.0,1.5}, sigma_b in {0.3,0.5,0.8}
mkdir -p ./experiments/logs/gulp_sweep

alpha_list=(0.8 1.0 1.2 1.5 2.0)
A_list=(0 0.1 0.2 0.3 0.5)
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

                logname=./experiments/logs/gulp_sweep/CIFAR10_GULP_a${safe_alpha}_A${safe_A}_mu${safe_mu}_s${safe_sigma}_seed42.log

                echo "Running GULP sweep: alpha=${alpha}, A=${A}, mu=${mu}, sigma_b=${sigma_b} -> ${logname}"

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
                    --gulp_alpha ${alpha} \
                    --gulp_amp ${A} \
                    --gulp_mu ${mu} \
                    --gulp_sigma ${sigma_b} \
                    --save_dir ./experiments \
                    >> "${logname}" 2>&1 &

                # throttle background launches to avoid oversubscribing a single GPU in interactive use
                sleep 0.5
            done
        done
    done
done

echo "Launched CIFAR-10 GULP sweep runs (backgrounded). Logs: ./experiments/logs/gulp_sweep/"
