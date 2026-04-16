import torch
import torch.nn as nn

class DepthBranch(nn.Module):

    def __init__(self,in_channels=512):
        super().__init__()

        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels,256,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,1,1)
        )

    def forward(self,x):
        depth_map = self.depth_head(x)
        return depth_map