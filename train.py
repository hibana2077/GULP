import argparse
import os
import json
import time
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dataset import get_dataloader
from models.model_factory import ModelFactory


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Disable bfloat16 to avoid compatibility issues
    os.environ['TORCH_DISABLE_BFLOAT16'] = '1'


def save_config(args: argparse.Namespace, save_dir: str) -> None:
    """Save experiment configuration"""
    config = vars(args)
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


def create_experiment_dir(args: argparse.Namespace) -> str:
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.dataset}_{args.model}_{args.activation}_{timestamp}_seed{args.seed}"
    exp_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    return {
        'loss': val_loss,
        'accuracy': val_acc
    }


def train_model(args: argparse.Namespace) -> Dict[str, Any]:
    """Main training function"""
    
    # Disable bfloat16 before importing torch modules
    os.environ['TORCH_DISABLE_BFLOAT16'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Disable mixed precision if on problematic hardware
    if device.type == 'cuda':
        try:
            # Test bfloat16 capability
            torch.tensor([1.0]).bfloat16()
        except:
            print("BFloat16 not supported, using float32")
            torch.backends.cudnn.allow_tf32 = False
    
    # Set seed
    set_seed(args.seed)
    
    # Create experiment directory
    exp_dir = create_experiment_dir(args)
    save_config(args, exp_dir)
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    train_loader, test_loader, num_classes = get_dataloader(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True
    )
    
    # Create model
    print(f"Creating {args.model} with {args.activation} activation...")
    
    # Prepare activation kwargs
    activation_kwargs = {}
    if args.activation.lower() == 'gulp':
        activation_kwargs = {
            'alpha': args.gulp_alpha,
            'bump_amp': args.gulp_amp,
            'mu': args.gulp_mu,
            'sigma': args.gulp_sigma,
            'n_bumps': args.gulp_n_bumps
        }
    
    if args.dataset.lower() in ['cifar10', 'cifar100']:
        model = ModelFactory.create_cifar_model(
            model_name=args.model,
            activation_name=args.activation,
            num_classes=num_classes,
            **activation_kwargs
        )
    else:
        model = ModelFactory.create_imagenet_model(
            model_name=args.model,
            activation_name=args.activation,
            num_classes=num_classes,
            pretrained=args.pretrained,
            **activation_kwargs
        )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'multistep':
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        scheduler = None
    
    # Training loop
    best_acc = 0.0
    results = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_acc': 0.0,
        'total_time': 0.0
    }
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_stats = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_stats = validate(model, test_loader, criterion, device)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Save results
        results['train_loss'].append(train_stats['loss'])
        results['train_acc'].append(train_stats['accuracy'])
        results['val_loss'].append(val_stats['loss'])
        results['val_acc'].append(val_stats['accuracy'])
        
        # Save best model
        if val_stats['accuracy'] > best_acc:
            best_acc = val_stats['accuracy']
            results['best_acc'] = best_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': args
            }, os.path.join(exp_dir, 'best_model.pth'))
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_stats['loss']:.4f}, "
              f"Train Acc: {train_stats['accuracy']:.2f}%, "
              f"Val Loss: {val_stats['loss']:.4f}, "
              f"Val Acc: {val_stats['accuracy']:.2f}%, "
              f"Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    results['total_time'] = total_time
    
    # Save final results
    results_path = os.path.join(exp_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'results': results,
        'args': args
    }, os.path.join(exp_dir, 'final_model.pth'))
    
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Results saved to: {exp_dir}")
    
    return results


def main():
    # Disable problematic features before any imports
    os.environ['TORCH_DISABLE_BFLOAT16'] = '1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    parser = argparse.ArgumentParser(description='Train models with different activations')
    
    # Dataset and model
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default='resnet18',
                       help='Model architecture (timm model name)')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'gelu', 'silu', 'swish', 'mish', 'gulp'],
                       help='Activation function')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam', 'adamw'],
                       help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'multistep', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30,
                       help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for StepLR')
    
    # GULP parameters
    parser.add_argument('--gulp_alpha', type=float, default=1.2,
                       help='GULP alpha parameter')
    parser.add_argument('--gulp_amp', type=float, default=0.25,
                       help='GULP amplitude parameter')
    parser.add_argument('--gulp_mu', type=float, default=1.0,
                       help='GULP mu parameter')
    parser.add_argument('--gulp_sigma', type=float, default=0.5,
                       help='GULP sigma parameter')
    parser.add_argument('--gulp_n_bumps', type=int, default=1,
                       help='GULP number of bumps')
    
    # Experiment settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./experiments',
                       help='Save directory')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    
    args = parser.parse_args()
    
    # Train model
    results = train_model(args)
    
    print(f"Final results: Best accuracy = {results['best_acc']:.2f}%")


if __name__ == '__main__':
    main()
