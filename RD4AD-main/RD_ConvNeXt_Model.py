"""
RD Model with Pure ConvNeXt Architecture
완전히 ConvNeXt 스타일로 재설계된 RD 모델
- LayerNorm 일관성 유지
- ConvNeXt 블록 구조 활용
- GELU 활성화 함수
- Depthwise Convolution 사용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Optional, List

# ConvNeXt path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ConvNext', 'Chap4'))
from ConvNext_V1 import convnext_tiny


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt Teacher Encoder - RD용"""
    def __init__(self, pretrained=True):
        super().__init__()

        # Pretrained ConvNeXt Tiny 로드
        self.backbone = convnext_tiny(pretrained=pretrained)

        # 필요한 stage만 사용
        self.stem = self.backbone.features[0]
        self.stage1 = self.backbone.features[1]  # 96 channels
        self.down1 = self.backbone.features[2]
        self.stage2 = self.backbone.features[3]  # 192 channels
        self.down2 = self.backbone.features[4]
        self.stage3 = self.backbone.features[5]  # 384 channels

        # Freeze encoder
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []

        x = self.stem(x)
        x = self.stage1(x)
        features.append(x.clone())  # 96 channels, 1/4 resolution

        x = self.down1(x)
        x = self.stage2(x)
        features.append(x.clone())  # 192 channels, 1/8 resolution

        x = self.down2(x)
        x = self.stage3(x)
        features.append(x.clone())  # 384 channels, 1/16 resolution

        return features


class ConvNeXtBNLayer(nn.Module):
    """ConvNeXt 스타일 Bottleneck Layer"""
    def __init__(self, channels=[96, 192, 384]):
        super().__init__()

        c1, c2, c3 = channels

        # ConvNeXt 스타일 처리 (GroupNorm으로 안정화)
        # Stage 1: 96 -> 384 (spatial reduction)
        self.process1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(min(32, c2), c2),  # 안전한 그룹 수
            nn.GELU(),
            nn.Conv2d(c2, c3, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(min(32, c3), c3),
            nn.GELU(),
        )

        # Stage 2: 192 -> 384 (spatial reduction)
        self.process2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(min(32, c3), c3),
            nn.GELU(),
        )

        # Stage 3: 384 -> 384 (no change)
        self.process3 = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, c3), c3),
            nn.GELU(),
        )

        # Fusion - 안전한 그룹 수로 단순화
        self.fusion = nn.Sequential(
            nn.Conv2d(c3 * 3, c3 * 2, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, c3 * 2), c3 * 2),
            nn.GELU(),
            nn.Conv2d(c3 * 2, c3 * 2, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(min(32, c3 * 2), c3 * 2),
            nn.GELU(),
        )

        self.output_channels = c3 * 2  # 768

    def forward(self, features):
        f1, f2, f3 = features

        # Process each scale
        p1 = self.process1(f1)  # 96 -> 384, 64x64 -> 16x16
        p2 = self.process2(f2)  # 192 -> 384, 32x32 -> 16x16
        p3 = self.process3(f3)  # 384 -> 384, 16x16 -> 16x16

        # Concatenate
        merged = torch.cat([p1, p2, p3], dim=1)  # [384*3, 16, 16]

        # Fusion
        output = self.fusion(merged)  # [768, 8, 8]

        return output


class ConvNeXtDecoder(nn.Module):
    """ConvNeXt 스타일 Decoder - GroupNorm 사용"""
    def __init__(self, input_channels=768, output_channels=[96, 192, 384]):
        super().__init__()

        c1, c2, c3 = output_channels

        # Upsampling blocks - GroupNorm으로 단순화
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(input_channels, c3, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(min(32, c3), c3),  # 안전한 그룹 수 설정
            nn.GELU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, c3), c3),
            nn.GELU(),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(min(32, c2), c2),
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, c2), c2),
            nn.GELU(),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(min(32, c1), c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, c1), c1),
            nn.GELU(),
        )

        # Output projections
        self.out1 = nn.Conv2d(c1, c1, kernel_size=1)
        self.out2 = nn.Conv2d(c2, c2, kernel_size=1)
        self.out3 = nn.Conv2d(c3, c3, kernel_size=1)

    def forward(self, x):
        # Progressive upsampling
        feat3 = self.up1(x)      # 768 -> 384, 8x8 -> 16x16
        feat2 = self.up2(feat3)  # 384 -> 192, 16x16 -> 32x32
        feat1 = self.up3(feat2)  # 192 -> 96, 32x32 -> 64x64

        # Output projections
        out1 = self.out1(feat1)  # 96 channels
        out2 = self.out2(feat2)  # 192 channels
        out3 = self.out3(feat3)  # 384 channels

        return [out1, out2, out3]


class RDConvNeXtModel(nn.Module):
    """Complete RD Model with ConvNeXt Architecture"""
    def __init__(self, pretrained=True):
        super().__init__()

        # Teacher Encoder (frozen)
        self.encoder = ConvNeXtEncoder(pretrained=pretrained)

        # Bottleneck Layer (trainable)
        self.bn_layer = ConvNeXtBNLayer(channels=[96, 192, 384])

        # Student Decoder (trainable)
        self.decoder = ConvNeXtDecoder(
            input_channels=768,
            output_channels=[96, 192, 384]
        )

    def forward(self, x):
        # Extract features with teacher
        with torch.no_grad():
            teacher_features = self.encoder(x)

        # Process through bottleneck
        bottleneck_features = self.bn_layer(teacher_features)

        # Reconstruct with student
        student_features = self.decoder(bottleneck_features)

        return teacher_features, student_features

    def freeze_encoder(self):
        """Freeze teacher encoder"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def get_trainable_params(self):
        """Get parameters for training (BN layer + Decoder)"""
        params = list(self.bn_layer.parameters()) + list(self.decoder.parameters())
        return params


def rd_convnext_model(pretrained=True):
    """Returns complete RD ConvNeXt model"""
    model = RDConvNeXtModel(pretrained=pretrained)
    model.freeze_encoder()
    return model


# Loss function for ConvNeXt features
def convnext_loss_function(teacher_features, student_features, normalize=True):
    """
    ConvNeXt 특징에 최적화된 loss function
    - Feature normalization 포함
    - Multi-scale loss
    """
    total_loss = 0
    weights = [1.0, 1.0, 1.0]  # 각 scale의 가중치

    for t_feat, s_feat, w in zip(teacher_features, student_features, weights):
        # Feature normalization (중요!)
        if normalize:
            t_feat = F.normalize(t_feat, p=2, dim=1)
            s_feat = F.normalize(s_feat, p=2, dim=1)

        # Cosine similarity loss
        cos_sim = F.cosine_similarity(t_feat, s_feat, dim=1)
        loss = torch.mean(1 - cos_sim)

        total_loss += w * loss

    return total_loss / sum(weights)