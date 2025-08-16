import torch
import torch.nn as nn
import timm
from typing import Dict, Any
import sys
import os

# Add the act module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'act'))

from gulp import GULP


def replace_activation_recursive(module: nn.Module, old_activation: type, new_activation: nn.Module) -> None:
    """
    Recursively replace all instances of old_activation with new_activation in a module
    """
    for name, child in module.named_children():
        if isinstance(child, old_activation):
            setattr(module, name, new_activation)
        else:
            replace_activation_recursive(child, old_activation, new_activation)


def get_activation_function(activation_name: str, **kwargs) -> nn.Module:
    """Get activation function by name"""
    activation_name = activation_name.lower()
    
    if activation_name == "relu":
        return nn.ReLU(inplace=True)
    elif activation_name == "gelu":
        return nn.GELU()
    elif activation_name == "silu" or activation_name == "swish":
        return nn.SiLU(inplace=True)
    elif activation_name == "mish":
        return nn.Mish(inplace=True)
    elif activation_name == "gulp":
        return GULP(**kwargs)
    else:
        raise ValueError(f"Unsupported activation: {activation_name}")


def create_model_with_activation(
    model_name: str,
    num_classes: int,
    activation_name: str,
    pretrained: bool = False,
    **activation_kwargs
) -> nn.Module:
    """
    Create a timm model and replace its activation functions
    
    Args:
        model_name: Name of the timm model
        num_classes: Number of output classes
        activation_name: Name of activation function to use
        pretrained: Whether to use pretrained weights
        **activation_kwargs: Additional arguments for the activation function
    
    Returns:
        Model with replaced activation functions
    """
    
    # Create base model
    model = timm.create_model(
        model_name,
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    # Get the new activation function
    new_activation = get_activation_function(activation_name, **activation_kwargs)
    
    # Replace activations based on model architecture
    if activation_name.lower() != "relu":  # ReLU is often the default, no need to replace
        # Common activation types to replace
        activation_types = [nn.ReLU, nn.GELU, nn.SiLU, nn.Mish]
        
        for act_type in activation_types:
            replace_activation_recursive(model, act_type, new_activation)
    
    return model


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model-specific configuration"""
    configs = {
        "resnet18": {
            "family": "resnet",
            "input_size": (3, 224, 224),
            "primary_activations": [nn.ReLU]
        },
        "resnet50": {
            "family": "resnet", 
            "input_size": (3, 224, 224),
            "primary_activations": [nn.ReLU]
        },
        "wide_resnet28_10": {
            "family": "wide_resnet",
            "input_size": (3, 32, 32),
            "primary_activations": [nn.ReLU]
        },
        "wide_resnet101_2": {
            "family": "wide_resnet",
            "input_size": (3, 224, 224), 
            "primary_activations": [nn.ReLU]
        },
        "vit_tiny_patch16_224": {
            "family": "vision_transformer",
            "input_size": (3, 224, 224),
            "primary_activations": [nn.GELU]
        },
        "vit_small_patch16_224": {
            "family": "vision_transformer", 
            "input_size": (3, 224, 224),
            "primary_activations": [nn.GELU]
        },
        "deit_tiny_patch16_224": {
            "family": "deit",
            "input_size": (3, 224, 224),
            "primary_activations": [nn.GELU]
        },
        "deit_small_patch16_224": {
            "family": "deit",
            "input_size": (3, 224, 224), 
            "primary_activations": [nn.GELU]
        }
    }
    
    return configs.get(model_name, {
        "family": "unknown",
        "input_size": (3, 224, 224),
        "primary_activations": [nn.ReLU]
    })


class ModelFactory:
    """Factory class for creating models with different activations"""
    
    @staticmethod
    def create_cifar_model(
        model_name: str,
        activation_name: str, 
        num_classes: int = 10,
        **activation_kwargs
    ) -> nn.Module:
        """Create model optimized for CIFAR datasets"""
        
        # For CIFAR, we typically use smaller input sizes
        if "resnet" in model_name.lower():
            # Use CIFAR-optimized ResNet if available
            if model_name == "resnet18":
                model_name = "resnet18"
            elif model_name == "resnet50": 
                model_name = "resnet50"
            # Note: Some timm models have CIFAR variants
        
        model = create_model_with_activation(
            model_name=model_name,
            num_classes=num_classes,
            activation_name=activation_name,
            pretrained=False,  # Usually no pretrained weights for CIFAR
            **activation_kwargs
        )
        
        # Modify first conv layer for CIFAR (32x32 instead of 224x224)
        if hasattr(model, 'conv1') and isinstance(model.conv1, nn.Conv2d):
            # Change stride and kernel size for CIFAR
            old_conv = model.conv1
            model.conv1 = nn.Conv2d(
                old_conv.in_channels,
                old_conv.out_channels, 
                kernel_size=3,
                stride=1,
                padding=1,
                bias=old_conv.bias is not None
            )
            
        # Remove max pooling for CIFAR if present
        if hasattr(model, 'maxpool'):
            model.maxpool = nn.Identity()
            
        return model
    
    @staticmethod
    def create_imagenet_model(
        model_name: str,
        activation_name: str,
        num_classes: int = 1000,
        pretrained: bool = True,
        **activation_kwargs
    ) -> nn.Module:
        """Create model for ImageNet"""
        
        return create_model_with_activation(
            model_name=model_name,
            num_classes=num_classes,
            activation_name=activation_name,
            pretrained=pretrained,
            **activation_kwargs
        )


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test CIFAR model
    model = ModelFactory.create_cifar_model(
        model_name="resnet18",
        activation_name="gulp",
        num_classes=10,
        alpha=1.2,
        bump_amp=0.25,
        mu=1.0,
        sigma=0.5
    )
    
    print(f"Model created: {type(model).__name__}")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
