"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import time
import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None,start_time=None,backbone_latency_list=None,encoder_latency_list=None,decoder_latency_list=None,encoder_encoder_latency_list=None):
        if self.training:
            x = self.backbone(x)
            x = self.encoder(x)
            x = self.decoder(x, targets)
        else:
            x = self.backbone(x)
            backbone_outtime=time.time()
            backbone_latency = (backbone_outtime - start_time) * 1000
            backbone_latency_list.append(backbone_latency)

            x = self.encoder(x)
            encoder_outtime=time.time()
            encoder_latency = (encoder_outtime - backbone_outtime) * 1000
            encoder_latency_list.append(encoder_latency)

            x = self.decoder(x, targets)
            decoder_outtime=time.time()
            decoder_latency = (decoder_outtime - encoder_outtime) * 1000
            decoder_latency_list.append(decoder_latency)

        out = x

        return out
        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
