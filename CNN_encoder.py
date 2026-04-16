import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(pretrained=True)

        self.features = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self,x):
        x = self.features(x)
        return x