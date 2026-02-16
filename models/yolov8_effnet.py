"""
YOLOv8 + EfficientNetV2 hybrid architecture.

Authors: Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .backbone import EfficientNetV2Backbone


class HybridYOLOv8(nn.Module):
    """
    Hybrid YOLOv8 with EfficientNetV2 backbone.
    
    Replaces the default CSP-Darknet53 backbone with EfficientNetV2-M
    while retaining the PAN-FPN neck and detection heads.
    
    Architecture:
        EfficientNetV2-M Backbone
            ├── P3 (80×80) → Adapter → 256 channels
            ├── P4 (40×40) → Adapter → 512 channels
            └── P5 (20×20) → Adapter → 1024 channels
                    ↓
        PAN-FPN Neck (bidirectional fusion)
                    ↓
        Detection Heads (anchor-free, decoupled)
    
    Args:
        num_classes: Number of detection classes
        variant: EfficientNet variant ('s', 'm', 'l')
        pretrained: Whether to use pretrained backbone
        in_channels: Input image channels
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        variant: str = 'm',
        pretrained: bool = True,
        in_channels: int = 3,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # EfficientNetV2 backbone
        self.backbone = EfficientNetV2Backbone(
            variant=variant,
            pretrained=pretrained,
            out_channels=256,
        )
        
        # Neck feature channels
        neck_channels = [256, 512, 1024]
        
        # Channel adapters to match YOLO neck expectations
        self.channel_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )
            for c in neck_channels
        ])
        
        # PAN-FPN Neck
        self.neck = PANFPNNeck(neck_channels)
        
        # Detection heads
        self.heads = nn.ModuleList([
            DetectionHead(c, num_classes) for c in neck_channels
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            List of detection outputs at each scale
        """
        # Extract backbone features
        features = self.backbone(x)
        
        # Adapt channels
        features = [adapter(f) for adapter, f in zip(self.channel_adapters, features)]
        
        # Apply neck
        features = self.neck(features)
        
        # Apply detection heads
        outputs = [head(f) for head, f in zip(self.heads, features)]
        
        return outputs


class PANFPNNeck(nn.Module):
    """
    Path Aggregation Network with Feature Pyramid Network.
    
    Performs bidirectional feature fusion:
    1. Top-down path: High-level semantics to low-level features
    2. Bottom-up path: Low-level details to high-level features
    """
    
    def __init__(self, channels: List[int]):
        super().__init__()
        
        # Top-down path (FPN)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.lateral_conv4 = ConvBNSiLU(channels[2], channels[1], 1)
        self.lateral_conv3 = ConvBNSiLU(channels[1], channels[0], 1)
        
        self.fpn_conv4 = C2f(channels[1] * 2, channels[1])
        self.fpn_conv3 = C2f(channels[0] * 2, channels[0])
        
        # Bottom-up path (PAN)
        self.downsample3 = ConvBNSiLU(channels[0], channels[0], 3, stride=2, padding=1)
        self.downsample4 = ConvBNSiLU(channels[1], channels[1], 3, stride=2, padding=1)
        
        self.pan_conv4 = C2f(channels[0] + channels[1], channels[1])
        self.pan_conv5 = C2f(channels[1] + channels[2], channels[2])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply PAN-FPN.
        
        Args:
            features: [P3, P4, P5] from backbone
            
        Returns:
            Fused features [P3, P4, P5]
        """
        p3, p4, p5 = features
        
        # Top-down FPN
        p5_up = self.upsample(self.lateral_conv4(p5))
        p4 = self.fpn_conv4(torch.cat([p4, p5_up], dim=1))
        
        p4_up = self.upsample(self.lateral_conv3(p4))
        p3 = self.fpn_conv3(torch.cat([p3, p4_up], dim=1))
        
        # Bottom-up PAN
        p3_down = self.downsample3(p3)
        p4 = self.pan_conv4(torch.cat([p4, p3_down], dim=1))
        
        p4_down = self.downsample4(p4)
        p5 = self.pan_conv5(torch.cat([p5, p4_down], dim=1))
        
        return [p3, p4, p5]


class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions (YOLOv8 style).
    """
    
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = ConvBNSiLU(c1, 2 * self.c, 1)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    
    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        self.cv1 = ConvBNSiLU(c1, c2, 3, padding=1)
        self.cv2 = ConvBNSiLU(c2, c2, 3, padding=1)
        self.add = shortcut and c1 == c2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class ConvBNSiLU(nn.Module):
    """Conv + BatchNorm + SiLU activation."""
    
    def __init__(self, c1: int, c2: int, k: int = 1, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DetectionHead(nn.Module):
    """
    Decoupled detection head (anchor-free).
    
    Separates classification and regression branches.
    """
    
    def __init__(self, in_channels: int, num_classes: int, reg_max: int = 16):
        super().__init__()
        
        hidden = max(in_channels // 4, 64)
        
        # Classification branch
        self.cls_conv = nn.Sequential(
            ConvBNSiLU(in_channels, hidden, 3, padding=1),
            ConvBNSiLU(hidden, hidden, 3, padding=1),
        )
        self.cls_pred = nn.Conv2d(hidden, num_classes, 1)
        
        # Regression branch
        self.reg_conv = nn.Sequential(
            ConvBNSiLU(in_channels, hidden, 3, padding=1),
            ConvBNSiLU(hidden, hidden, 3, padding=1),
        )
        self.reg_pred = nn.Conv2d(hidden, 4 * reg_max, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (classification_logits, regression_output)
        """
        cls_feat = self.cls_conv(x)
        cls_out = self.cls_pred(cls_feat)
        
        reg_feat = self.reg_conv(x)
        reg_out = self.reg_pred(reg_feat)
        
        return cls_out, reg_out


if __name__ == "__main__":
    print("Testing HybridYOLOv8...")
    
    model = HybridYOLOv8(num_classes=4, variant='m', pretrained=False)
    
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    
    print(f"\nInput: {x.shape}")
    print(f"Outputs:")
    for i, (cls_out, reg_out) in enumerate(outputs):
        print(f"  Scale {i}: cls={cls_out.shape}, reg={reg_out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    
    print("\nTest passed!")
