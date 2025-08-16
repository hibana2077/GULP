#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=20GB           
#PBS -l walltime=14:30:00  
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

# Run complete Table 3 experiments with multiple seeds
echo "Starting complete Table 3 experiments..."

# Define seeds
SEEDS=(42 123 456 789 999)

# CIFAR-10 experiments
echo "Running CIFAR-10 experiments..."
for seed in "${SEEDS[@]}"; do
    echo "Seed: $seed"
    
    # ReLU
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
        --seed $seed \
        --save_dir ./experiments \
        >> CIFAR10_ReLU_seed${seed}.log 2>&1
    
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
        --seed $seed \
        --save_dir ./experiments \
        >> CIFAR10_GELU_seed${seed}.log 2>&1
    
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
        --seed $seed \
        --save_dir ./experiments \
        >> CIFAR10_SiLU_seed${seed}.log 2>&1
    
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
        --seed $seed \
        --gulp_alpha 1.2 \
        --gulp_amp 0.25 \
        --gulp_mu 1.0 \
        --gulp_sigma 0.5 \
        --save_dir ./experiments \
        >> CIFAR10_GULP_seed${seed}.log 2>&1
done

# CIFAR-100 experiments
echo "Running CIFAR-100 experiments..."
for seed in "${SEEDS[@]}"; do
    echo "Seed: $seed"
    
    # ReLU
    python3 train.py \
        --dataset cifar100 \
        --model resnet50.fb_swsl_ig1b_ft_in1k \
        --activation relu \
        --epochs 100 \
        --batch_size 128 \
        --lr 0.001 \
        --optimizer adamw \
        --momentum 0.9 \
        --weight_decay 5e-4 \
        --scheduler cosine \
        --seed $seed \
        --save_dir ./experiments \
        >> CIFAR100_ReLU_seed${seed}.log 2>&1
    
    # GELU
    python3 train.py \
        --dataset cifar100 \
        --model resnet50.fb_swsl_ig1b_ft_in1k \
        --activation gelu \
        --epochs 100 \
        --batch_size 128 \
        --lr 0.001 \
        --optimizer adamw \
        --momentum 0.9 \
        --weight_decay 5e-4 \
        --scheduler cosine \
        --seed $seed \
        --save_dir ./experiments \
        >> CIFAR100_GELU_seed${seed}.log 2>&1
    
    # SiLU
    python3 train.py \
        --dataset cifar100 \
        --model resnet50.fb_swsl_ig1b_ft_in1k \
        --activation silu \
        --epochs 100 \
        --batch_size 128 \
        --lr 0.001 \
        --optimizer adamw \
        --momentum 0.9 \
        --weight_decay 5e-4 \
        --scheduler cosine \
        --seed $seed \
        --save_dir ./experiments \
        >> CIFAR100_SiLU_seed${seed}.log 2>&1
    
    # GULP
    python3 train.py \
        --dataset cifar100 \
        --model resnet50.fb_swsl_ig1b_ft_in1k \
        --activation gulp \
        --epochs 100 \
        --batch_size 128 \
        --lr 0.001 \
        --optimizer adamw \
        --momentum 0.9 \
        --weight_decay 5e-4 \
        --scheduler cosine \
        --seed $seed \
        --gulp_alpha 1.2 \
        --gulp_amp 0.25 \
        --gulp_mu 1.0 \
        --gulp_sigma 0.5 \
        --save_dir ./experiments \
        >> CIFAR100_GULP_seed${seed}.log 2>&1
done

echo "All Table 3 experiments completed!"

# Generate results summary
python3 -c "
import os
import json
import pandas as pd
from pathlib import Path

# Collect all results
results = []
exp_dir = Path('./experiments')

for result_file in exp_dir.glob('*/results.json'):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract info from directory name
        dir_name = result_file.parent.name
        parts = dir_name.split('_')
        if len(parts) >= 4:
            dataset = parts[0]
            model = parts[1]
            activation = parts[2]
            seed = parts[-1].replace('seed', '')
            
            results.append({
                'dataset': dataset,
                'model': model, 
                'activation': activation,
                'seed': int(seed),
                'best_acc': data['best_acc'],
                'final_train_acc': data['train_acc'][-1],
                'final_val_acc': data['val_acc'][-1],
                'total_time': data['total_time']
            })
    except:
        continue

if results:
    df = pd.DataFrame(results)
    df.to_csv('./experiments/table3_results.csv', index=False)
    
    # Calculate summary statistics
    summary = df.groupby(['dataset', 'model', 'activation'])['best_acc'].agg(['mean', 'std']).reset_index()
    summary.to_csv('./experiments/table3_summary.csv', index=False)
    
    print('Results saved to ./experiments/table3_results.csv')
    print('Summary saved to ./experiments/table3_summary.csv')
else:
    print('No results found')
"
