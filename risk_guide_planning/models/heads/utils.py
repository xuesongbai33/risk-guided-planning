# Copyright (c) OpenMMLab. All rights reserved.
"""
Utility modules for Risk Cognition Head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation module.
    
    Args:
        in_channels (int): Number of input channels.
        cond_dim (int): Dimension of condition vector.
        hidden_dim (int, optional): Dimension of hidden layer in condition mapping network.
            Defaults to None, which uses cond_dim * 2.
    """
    
    def __init__(self, in_channels, cond_dim, hidden_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        
        # add condition mapping network
        if hidden_dim is None:
            hidden_dim = cond_dim * 2
            
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels * 2)  # output gamma and beta
        )
        
        # init gamma≈1, beta≈0 (identity mapping)
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)
        self.film_gen[-1].bias.data[:in_channels] = 1.0  # init gamma to 1
    
    def forward(self, features, condition):
        """
        Forward pass of FiLM modulation.
        
        Args:
            features (Tensor): Input features with shape [B, C, H, W].
            condition (Tensor): Condition vector with shape [B, D] (raw condition).
            
        Returns:
            Tensor: Modulated features with shape [B, C, H, W].
        """
        B, C, H, W = features.shape
        
        # generate FiLM params
        film_params = self.film_gen(condition)  # [B, 2*C]
        gamma, beta = film_params.chunk(2, dim=1)  # each [B, C]
        
        # reshape and apply
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        
        return features * gamma + beta


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rates (list): List of dilation rates for atrous convolutions.
    """
    
    def __init__(self, in_channels, out_channels, rates):
        super().__init__()
        self.rates = rates
        self.convs = nn.ModuleList()
        
        # 1x1 convolution branch
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolution branches with different rates
        for rate in rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    padding=rate, 
                    dilation=rate, 
                    bias=False
                ),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion convolution
        num_branches = len(self.convs) + 1  # including global pooling branch
        self.fusion = nn.Sequential(
            nn.Conv2d(
                num_branches * out_channels, 
                out_channels, 
                kernel_size=1, 
                bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of ASPP module.
        
        Args:
            x (Tensor): Input features with shape [B, C, H, W].
            
        Returns:
            Tensor: Output features with shape [B, out_channels, H, W].
        """
        outputs = []
        
        # Compute all convolution branches
        for conv in self.convs:
            outputs.append(conv(x))
        
        # Compute global average pooling branch
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(
            global_feat, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        outputs.append(global_feat)
        
        # Concatenate all branches and fuse
        fused = torch.cat(outputs, dim=1)
        return self.fusion(fused)
