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


# GULP
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
    --gulp_alpha 2.0 \
    --gulp_amp 0.2 \
    --gulp_mu 0.5 \
    --gulp_sigma 0.8 \
    --save_dir ./experiments \
    >> CIFAR10_GULP_seed42.log 2>&1 &

echo "Launched CIFAR-10 GULP sweep runs (backgrounded). Logs: ./experiments/logs/gulp_sweep/"
