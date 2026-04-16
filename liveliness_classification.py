import torch
import torch.nn as nn
import torchvision.models as models


class DepthBranch(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()

        self.depth = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        return self.depth(x)


class LivenessNet(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # remove last pooling + fc
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])

        self.depth_branch = DepthBranch(512)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,1)
        )

    def forward(self, x):

        features = self.encoder(x)

        depth_map = self.depth_branch(features)

        pooled = self.pool(features)

        pooled = pooled.view(pooled.size(0), -1)

        pred = self.classifier(pooled)

        return pred, depth_map
