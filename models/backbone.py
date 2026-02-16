"""
Backbone networks for feature extraction.

Authors: Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang
"""

import torch
import torch.nn as nn
from typing import List, Optional

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class EfficientNetV2Backbone(nn.Module):
    """
    EfficientNetV2 backbone for multi-scale feature extraction.
    
    Extracts features at P3, P4, P5 levels corresponding to 
    strides 8, 16, 32 for 640x640 input.
    
    Args:
        variant: Model variant ('s', 'm', 'l')
        pretrained: Whether to use ImageNet pretrained weights
        out_channels: Output channels after adaptation
        freeze_bn: Whether to freeze batch normalization
        
    Output feature sizes for 640x640 input:
        - P3: 80x80
        - P4: 40x40
        - P5: 20x20
    """
    
    VARIANTS = {
        's': 'tf_efficientnetv2_s',
        'm': 'tf_efficientnetv2_m',
        'l': 'tf_efficientnetv2_l',
    }
    
    CHANNELS = {
        's': [64, 160, 256],
        'm': [80, 176, 512],
        'l': [96, 224, 640],
    }
    
    def __init__(
        self,
        variant: str = 'm',
        pretrained: bool = True,
        out_channels: int = 256,
        freeze_bn: bool = False,
    ):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for EfficientNetV2 backbone")
        
        self.variant = variant
        self.out_channels = out_channels
        
        # Get model name
        model_name = self.VARIANTS.get(variant)
        if model_name is None:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(self.VARIANTS.keys())}")
        
        # Load backbone with feature extraction
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[2, 4, 6],  # P3, P4, P5 stages
        )
        
        # Get channel dimensions
        in_channels = self.CHANNELS[variant]
        
        # Feature adaptation layers
        self.adapters = nn.ModuleList([
            self._make_adapter(c, out_channels) for c in in_channels
        ])
        
        if freeze_bn:
            self._freeze_bn()
    
    def _make_adapter(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create feature adaptation module."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
    
    def _freeze_bn(self):
        """Freeze batch normalization layers."""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            List of feature tensors [P3, P4, P5]
        """
        # Extract backbone features
        features = self.backbone(x)
        
        # Apply adapters
        adapted = [adapter(f) for adapter, f in zip(self.adapters, features)]
        
        return adapted
    
    @property
    def output_channels(self) -> List[int]:
        """Get output channel dimensions."""
        return [self.out_channels] * 3


class CSPDarknet53(nn.Module):
    """
    CSP-Darknet53 backbone (YOLO default).
    
    For comparison with EfficientNetV2.
    """
    
    def __init__(self, out_channels: int = 256):
        super().__init__()
        # This is a placeholder - in practice, use YOLO's implementation
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError("Use YOLO's native implementation")


if __name__ == "__main__":
    print("Testing EfficientNetV2 backbone...")
    
    # Test different variants
    for variant in ['s', 'm', 'l']:
        print(f"\nVariant: {variant}")
        backbone = EfficientNetV2Backbone(variant=variant, pretrained=False)
        
        x = torch.randn(1, 3, 640, 640)
        features = backbone(x)
        
        for i, f in enumerate(features):
            print(f"  P{i+3}: {f.shape}")
    
    print("\nBackbone tests passed!")
