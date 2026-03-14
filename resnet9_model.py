import torch
import torch.nn as nn
import torchvision.models as models

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.base_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            *list(resnet.layer1),
            *list(resnet.layer2)
        )
        self.residual_block = nn.Sequential(*list(resnet.layer3), *list(resnet.layer4))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, num_classes)
        )

    def forward(self, xb):
        out = self.base_layers(xb)
        out = self.residual_block(out)
        out = self.classifier(out)
        return out
