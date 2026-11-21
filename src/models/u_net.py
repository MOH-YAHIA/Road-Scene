import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNet50Encoder(nn.Module):
    def __init__(self, n_freeze_layers=0):
        super().__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Freeze the first n modules in this ordered list.
        self.backbone = [
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

        for layer in self.backbone[:n_freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        stem = self.stem(x)  # 64, H/2, W/2
        x = self.maxpool(stem)  # 64, H/4, W/4
        layer1 = self.layer1(x)  # 256, H/4, W/4
        layer2 = self.layer2(layer1)  # 512, H/8, W/8
        layer3 = self.layer3(layer2)  # 1024, H/16, W/16
        layer4 = self.layer4(layer3)  # 2048, H/32, W/32

        return stem, layer1, layer2, layer3, layer4


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Concatenated channels = decoder channels + skip channels.
        self.block1 = ConvBlock(2048 + 1024, 512)
        self.block2 = ConvBlock(512 + 512, 256)
        self.block3 = ConvBlock(256 + 256, 128)
        self.block4 = ConvBlock(128 + 64, 64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, features, output_size):
        stem, layer1, layer2, layer3, layer4 = features

        x = layer4

        # 1/32 -> 1/16
        x = F.interpolate(
            x,
            size=layer3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, layer3], dim=1)
        x = self.block1(x)

        # 1/16 -> 1/8
        x = F.interpolate(
            x,
            size=layer2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, layer2], dim=1)
        x = self.block2(x)

        # 1/8 -> 1/4
        x = F.interpolate(
            x,
            size=layer1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, layer1], dim=1)
        x = self.block3(x)

        # 1/4 -> 1/2
        x = F.interpolate(
            x,
            size=stem.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, stem], dim=1)
        x = self.block4(x)

        # 1/2 -> original input resolution
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return self.classifier(x)


class UNetSegmentationModel(nn.Module):
    def __init__(
        self,
        num_classes,
        n_freeze_layers=0,
    ):
        super().__init__()

        self.encoder = ResNet50Encoder(n_freeze_layers=n_freeze_layers)
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        output_size = x.shape[-2:]

        features = self.encoder(x)
        logits = self.decoder(features, output_size)

        return logits
