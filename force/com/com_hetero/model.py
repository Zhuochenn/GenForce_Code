import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gru import ConvGRU
from extractor import FeatureEncoder, ResidualBlock


class TemporalForce(nn.Module):  
    def __init__(self, out_dim=3, feature_dim=128, hidden_dim=128, num_layers=1, dropout=0.0):  
        super(TemporalForce, self).__init__()  
        self.out_dim = out_dim  
        self.feature_dim = feature_dim  
        self.hidden_dim = hidden_dim    #fixed to be 128
        self.num_layers = num_layers  

        # Backbone for feature extraction 
        self.base_network = FeatureEncoder(output_dim=self.feature_dim, norm_fn='instance', dropout=dropout)    # s,b,128,32,32

        # ConvGRU module to process spatiotemporal information  
        self.convgru = ConvGRU(input_dim=self.feature_dim, hidden_dim=self.hidden_dim, num_layers=num_layers)    # s,b,128,32,32

        # Post processing layer
        self.norm_fn = "batch"
        layer1 = ResidualBlock(128,256, self.norm_fn, stride=2)  # sxb,256,16,16
        layer2 = ResidualBlock(256,512, self.norm_fn, stride=2)  # sxb,512,8,8
        layer3 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        layer_post_processing  =  (layer1, layer2, layer3)
        self.post_processing = nn.Sequential(*layer_post_processing)

        # Regression head for mapping GRU outputs to force predictions  
        self.reg_layer = nn.Sequential(     
            nn.Linear(512, self.out_dim),  
            nn.Sigmoid()  
        )  


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, src_imgs):  
        """  
        Args:  
            src_imgs: Input tensor of shape (S, B, C, H, W) - sequence of images  
            src_forces: Ground-truth forces of shape (S, B, 3)  
        Returns:  
            reg_loss: Loss value computed using predicted forces and ground-truth forces  
        """  
        s, b, c, h, w = src_imgs.shape  

        # Feature extraction with backbone  
        features = self.base_network(src_imgs.reshape(-1, c, h, w))  # Shape: (s * b, feature_dim, 32, 32)  
        features = features.view(s, b, self.feature_dim, 32, 32)  # Reshape to include spatial dimensions  

        # Forward through ConvGRU  
        outputs, _ = self.convgru(features)  # Shape: (s, b, hidden_dim, 32, 32)  

        # Forward through Post processing  
        outputs = outputs.view(-1, self.feature_dim, 32, 32)   # Shape: (s * b, hidden_dim, 32, 32)  
        outputs = self.post_processing(outputs)  # Shape: (s * b, 512, 1, 1)  
        # Forward through Reg Layer  
        outputs =  outputs.view(-1, 512)  # Shape: (s * b, 512)  
        outputs= self.reg_layer(outputs)  # Shape: (s * b, 3) 
        outputs =  outputs.view(s,b,-1)  # Shape: (s, b, 3)  

        return outputs

    def get_parameters(self, initial_lr=1.0):  
        """  
        Return model parameters grouped with corresponding learning rates.  
        """  
        params = [  
            {'params': self.base_network.parameters(), 'lr': 1.0 * initial_lr},  
            {'params': self.convgru.parameters(), 'lr': 1.0 * initial_lr},  
            {'params': self.post_processing.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.reg_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]  
        return params  