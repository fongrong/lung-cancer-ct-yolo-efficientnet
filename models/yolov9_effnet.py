"""
YOLOv9 + EfficientNetV2 hybrid architecture with PGI module.

Authors: Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from .backbone import EfficientNetV2Backbone
from .yolov8_effnet import PANFPNNeck, DetectionHead, ConvBNSiLU


class HybridYOLOv9(nn.Module):
    """
    Hybrid YOLOv9 with EfficientNetV2 backbone and PGI module.
    
    Args:
        num_classes: Number of detection classes
        variant: EfficientNet variant ('s', 'm', 'l')
        pretrained: Whether to use pretrained backbone
        use_pgi: Whether to use PGI during training
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        variant: str = 'm',
        pretrained: bool = True,
        use_pgi: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_pgi = use_pgi
        
        # EfficientNetV2 backbone
        self.backbone = EfficientNetV2Backbone(
            variant=variant,
            pretrained=pretrained,
            out_channels=256,
        )
        
        # Channel dimensions
        neck_channels = [256, 512, 1024]
        
        # Channel adapters
        self.channel_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )
            for c in neck_channels
        ])
        
        # PGI modules for each scale
        if use_pgi:
            self.pgi_modules = nn.ModuleList([
                PGIModule(c) for c in neck_channels
            ])
        
        # GELAN-style feature processing
        self.gelan = GELANModule(neck_channels)
        
        # PAN-FPN Neck
        self.neck = PANFPNNeck(neck_channels)
        
        # Detection heads
        self.heads = nn.ModuleList([
            DetectionHead(c, num_classes) for c in neck_channels
        ])
    
    def forward(self, x: torch.Tensor):
        """Forward pass."""
        # Extract backbone features
        features = self.backbone(x)
        
        # Adapt channels
        features = [adapter(f) for adapter, f in zip(self.channel_adapters, features)]
        
        # Apply PGI (training only)
        aux_outputs = None
        if self.training and self.use_pgi:
            aux_features = [pgi(f) for pgi, f in zip(self.pgi_modules, features)]
            aux_outputs = aux_features
        
        # Apply GELAN processing
        features = self.gelan(features)
        
        # Apply neck
        features = self.neck(features)
        
        # Apply detection heads
        outputs = [head(f) for head, f in zip(self.heads, features)]
        
        if self.training and self.use_pgi and aux_outputs is not None:
            return outputs, aux_outputs
        
        return outputs


class PGIModule(nn.Module):
    """Programmable Gradient Information module."""
    
    def __init__(self, channels: int, num_blocks: int = 3):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out) + out
        return out


class GELANModule(nn.Module):
    """Generalized Efficient Layer Aggregation Network module."""
    
    def __init__(self, channels: List[int]):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            GELANBlock(c) for c in channels
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [block(f) for block, f in zip(self.blocks, features)]


class GELANBlock(nn.Module):
    """Single GELAN block."""
    
    def __init__(self, channels: int, num_layers: int = 4):
        super().__init__()
        
        hidden = channels // 2
        
        self.split_conv = ConvBNSiLU(channels, hidden * 2, 1)
        
        self.layers = nn.ModuleList([
            ConvBNSiLU(hidden, hidden, 3, padding=1)
            for _ in range(num_layers)
        ])
        
        self.fuse = ConvBNSiLU(hidden * (num_layers + 2), channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = self.split_conv(x)
        chunks = list(split.chunk(2, dim=1))
        
        for layer in self.layers:
            chunks.append(layer(chunks[-1]))
        
        return self.fuse(torch.cat(chunks, dim=1))
