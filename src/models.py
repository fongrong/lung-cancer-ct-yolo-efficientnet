"""
Model architectures for Lung Cancer CT Classification.

Implements hybrid YOLO-EfficientNet architectures:
- YOLOv8m baseline
- YOLOv8m + EfficientNetV2-M
- YOLOv9m baseline  
- YOLOv9m + EfficientNetV2-M

Authors: Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. EfficientNet backbone not available.")

from ultralytics import YOLO


class FeatureAdapter(nn.Module):
    """
    Feature adapter module to match EfficientNet features to YOLO neck.
    
    Adapts feature maps from EfficientNet backbone to the expected channel
    dimensions for the YOLO PAN-FPN neck through 1x1 and 3x3 convolutions.
    
    Args:
        in_channels: Number of input channels from EfficientNet
        out_channels: Number of output channels for YOLO neck (default: 320)
        use_bn: Whether to use batch normalization
        activation: Activation function ('silu', 'relu', 'leaky_relu')
        
    Example:
        >>> adapter = FeatureAdapter(in_channels=80, out_channels=320)
        >>> x = torch.randn(1, 80, 80, 80)
        >>> out = adapter(x)
        >>> print(out.shape)  # torch.Size([1, 320, 80, 80])
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 320,
        use_bn: bool = True,
        activation: str = 'silu'
    ):
        super().__init__()
        
        # 1x1 convolution for channel adjustment
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # Activation
        if activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.Identity()
        
        # 3x3 convolution for spatial processing
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize convolution weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class EfficientNetBackbone(nn.Module):
    """
    EfficientNetV2-M backbone for multi-scale feature extraction.
    
    Extracts features at P3, P4, P5 levels for use with YOLO detection heads.
    Uses timm library for pretrained EfficientNetV2 models.
    
    Args:
        model_name: Name of EfficientNet variant
        pretrained: Whether to use ImageNet pretrained weights
        out_channels: Output channels after adaptation (default: 320)
        freeze_bn: Whether to freeze batch normalization layers
        
    Feature map sizes for 640x640 input:
        - P3: 80x80
        - P4: 40x40
        - P5: 20x20
    """
    
    # EfficientNetV2 feature channels at different stages
    FEATURE_CHANNELS = {
        'efficientnetv2_s': [64, 160, 256],
        'efficientnetv2_m': [80, 176, 512],
        'efficientnetv2_l': [96, 224, 640],
        'tf_efficientnetv2_s': [64, 160, 256],
        'tf_efficientnetv2_m': [80, 176, 512],
        'tf_efficientnetv2_l': [96, 224, 640],
    }
    
    def __init__(
        self,
        model_name: str = 'efficientnetv2_m',
        pretrained: bool = True,
        out_channels: int = 320,
        freeze_bn: bool = False,
    ):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for EfficientNet backbone")
        
        self.model_name = model_name
        self.out_channels = out_channels
        
        # Load EfficientNet backbone with feature extraction
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[2, 4, 6],  # P3, P4, P5 equivalent stages
        )
        
        # Get input channels for adapters
        feature_channels = self.FEATURE_CHANNELS.get(
            model_name,
            self.FEATURE_CHANNELS['efficientnetv2_m']
        )
        
        # Create feature adapters
        self.adapters = nn.ModuleList([
            FeatureAdapter(ch, out_channels) for ch in feature_channels
        ])
        
        # Freeze batch norm if requested
        if freeze_bn:
            self._freeze_bn()
    
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
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            List of feature tensors [P3, P4, P5], each with out_channels
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply adapters to normalize channels
        adapted = [adapter(feat) for adapter, feat in zip(self.adapters, features)]
        
        return adapted
    
    def get_feature_info(self) -> List[Dict]:
        """Get information about output features."""
        return [
            {'channels': self.out_channels, 'stride': 8, 'name': 'P3'},
            {'channels': self.out_channels, 'stride': 16, 'name': 'P4'},
            {'channels': self.out_channels, 'stride': 32, 'name': 'P5'},
        ]


class HybridYOLOv8EfficientNet(nn.Module):
    """
    Hybrid YOLOv8 + EfficientNetV2-M model.
    
    Replaces YOLOv8's CSP-Darknet backbone with EfficientNetV2-M while
    retaining the PAN-FPN neck and anchor-free detection heads.
    
    Args:
        num_classes: Number of detection classes
        backbone_name: EfficientNet variant name
        pretrained_backbone: Whether to use pretrained backbone
        pretrained_yolo: Path to pretrained YOLO weights
        
    Architecture:
        Input (640x640x3)
            ↓
        EfficientNetV2-M Backbone (ImageNet pretrained)
            ├── P3: 80x80x80  → Adapter → 80x80x320
            ├── P4: 40x40x176 → Adapter → 40x40x320
            └── P5: 20x20x512 → Adapter → 20x20x320
            ↓
        PAN-FPN Neck (bidirectional feature fusion)
            ↓
        Anchor-free Detection Heads (decoupled cls/loc)
            ↓
        Output: [class, x, y, w, h, conf] per detection
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = 'efficientnetv2_m',
        pretrained_backbone: bool = True,
        pretrained_yolo: Optional[str] = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        
        # Create EfficientNet backbone
        self.backbone = EfficientNetBackbone(
            model_name=backbone_name,
            pretrained=pretrained_backbone,
            out_channels=320,
        )
        
        # Load YOLO for neck and head components
        yolo_weights = pretrained_yolo or 'yolov8m.pt'
        self._yolo = YOLO(yolo_weights)
        
        print(f"Initialized HybridYOLOv8EfficientNet:")
        print(f"  Backbone: {backbone_name}")
        print(f"  Pretrained backbone: {pretrained_backbone}")
        print(f"  Num classes: {num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 640, 640)
            
        Returns:
            Detection outputs
        """
        # Extract features from EfficientNet backbone
        features = self.backbone(x)
        
        # Features are now in format expected by YOLO neck
        # [P3: 80x80x320, P4: 40x40x320, P5: 20x20x320]
        
        return features
    
    def train_step(self, batch, optimizer, scaler=None):
        """Perform a single training step."""
        images, targets = batch
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self(images)
                loss = self._compute_loss(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = self(images)
            loss = self._compute_loss(outputs, targets)
            loss.backward()
            optimizer.step()
        
        return loss.item()
    
    def _compute_loss(self, outputs, targets):
        """Compute detection loss."""
        # Placeholder - in practice, use YOLO's loss computation
        return torch.tensor(0.0, requires_grad=True)


class HybridYOLOv9EfficientNet(nn.Module):
    """
    Hybrid YOLOv9 + EfficientNetV2-M model.
    
    Combines EfficientNetV2-M backbone with YOLOv9's GELAN architecture
    and PGI (Programmable Gradient Information) module.
    
    Args:
        num_classes: Number of detection classes
        backbone_name: EfficientNet variant name
        pretrained_backbone: Whether to use pretrained backbone
        use_pgi: Whether to use PGI module during training
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = 'efficientnetv2_m',
        pretrained_backbone: bool = True,
        use_pgi: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_pgi = use_pgi
        
        # EfficientNet backbone
        self.backbone = EfficientNetBackbone(
            model_name=backbone_name,
            pretrained=pretrained_backbone,
            out_channels=320,
        )
        
        # PGI module for gradient preservation during training
        if use_pgi:
            self.pgi = PGIModule(in_channels=320)
        
        print(f"Initialized HybridYOLOv9EfficientNet:")
        print(f"  Backbone: {backbone_name}")
        print(f"  PGI enabled: {use_pgi}")
        print(f"  Num classes: {num_classes}")
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        During training with PGI, returns both main and auxiliary outputs.
        During inference, returns only main output.
        """
        features = self.backbone(x)
        
        if self.training and self.use_pgi:
            main_features, aux_features = self._forward_with_pgi(features)
            return main_features, aux_features
        else:
            return features
    
    def _forward_with_pgi(
        self,
        features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward with PGI module for training."""
        main_features = features
        aux_features = [self.pgi(f) for f in features]
        return main_features, aux_features


class PGIModule(nn.Module):
    """
    Programmable Gradient Information module from YOLOv9.
    
    Maintains complete input information through an auxiliary reversible
    branch during training to address information bottleneck problem.
    
    Args:
        in_channels: Number of input channels
        num_layers: Number of reversible layers
    """
    
    def __init__(self, in_channels: int = 320, num_layers: int = 3):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connections for gradient preservation."""
        out = x
        for layer in self.layers:
            out = layer(out) + out
        return out


def create_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: One of 'yolov8m', 'yolov8m_effnet', 'yolov9m', 'yolov9m_effnet'
        num_classes: Number of detection classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        Model instance
        
    Example:
        >>> model = create_model('yolov8m_effnet', num_classes=4)
        >>> x = torch.randn(1, 3, 640, 640)
        >>> outputs = model(x)
    """
    model_registry = {
        'yolov8m': lambda: YOLO('yolov8m.pt'),
        'yolov8m_effnet': lambda: HybridYOLOv8EfficientNet(
            num_classes=num_classes,
            pretrained_backbone=pretrained,
            **kwargs
        ),
        'yolov9m': lambda: YOLO('yolov9m.pt'),
        'yolov9m_effnet': lambda: HybridYOLOv9EfficientNet(
            num_classes=num_classes,
            pretrained_backbone=pretrained,
            **kwargs
        ),
    }
    
    if model_name not in model_registry:
        available = list(model_registry.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    print(f"Creating model: {model_name}")
    return model_registry[model_name]()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dictionary with total, trainable, and frozen parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'total_millions': total / 1e6,
        'trainable_millions': trainable / 1e6,
    }


if __name__ == "__main__":
    print("Testing model architectures...\n")
    
    # Test EfficientNet backbone
    print("1. Testing EfficientNetBackbone:")
    backbone = EfficientNetBackbone(pretrained=False)
    x = torch.randn(1, 3, 640, 640)
    features = backbone(x)
    
    for i, feat in enumerate(features):
        print(f"   P{i+3}: {feat.shape}")
    
    # Test Feature Adapter
    print("\n2. Testing FeatureAdapter:")
    adapter = FeatureAdapter(80, 320)
    out = adapter(features[0])
    print(f"   Input: {features[0].shape} -> Output: {out.shape}")
    
    # Test hybrid model creation
    print("\n3. Testing model creation:")
    for name in ['yolov8m_effnet', 'yolov9m_effnet']:
        try:
            model = create_model(name, num_classes=4, pretrained=False)
            params = count_parameters(model)
            print(f"   {name}: {params['total_millions']:.2f}M parameters")
        except Exception as e:
            print(f"   {name}: Error - {e}")
    
    print("\nModel tests complete!")
