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


class Encoder(nn.Module):
    def __init__(self, n_freeze_layers):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Freeze the first n modules in this ordered list.
        self.layers = [
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

        for layer in self.layers[:n_freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        # B, 3, H, W
        stem_o = self.stem(x)  # B, 64, H/2, W/2
        pool_o = self.pool(stem_o)  # B, 64, H/4, W/4
        layer1_o = self.layer1(pool_o)  # B, 256, H/4, W/4
        layer2_o = self.layer2(layer1_o)  # B, 512, H/8, W/8
        layer3_o = self.layer3(layer2_o)  # B, 1024, H/16, W/16
        layer4_o = self.layer4(layer3_o)  # B, 2048, H/32, W/32

        return stem_o, pool_o, layer1_o, layer2_o, layer3_o, layer4_o


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.block1 = ConvBlock(2048 + 1024, 1024)  # layer4 undo
        self.block2 = ConvBlock(1024 + 512, 512)  # layer3 undo
        self.block3 = ConvBlock(512 + 256, 256)  # layer2 undo
        self.block4 = ConvBlock(256 + 64, 128)  # layer1 undo

        self.classifier = nn.Conv2d(  # stem undo
            128,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        layers_o,
        output_size,
    ):
        stem_o, _, layer1_o, layer2_o, layer3_o, layer4_o = layers_o

        # layer4_o
        # B, 2048, H/32, W/32
        x = F.interpolate(
            layer4_o,
            size=layer3_o.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # B, 2048, H/16, W/16
        x = torch.cat([x, layer3_o], dim=1)
        # B, 2048+1024, H/16, W/16
        x = self.block1(x)
        # B, 1024, H/16, W/16

        x = F.interpolate(
            x,
            size=layer2_o.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # B, 1024, H/8, W/8
        x = torch.cat([x, layer2_o], dim=1)
        # B, 1024+512, H/8, W/8
        x = self.block2(x)
        # B, 512, H/8, W/8

        x = F.interpolate(
            x,
            size=layer1_o.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # B, 512, H/4, W/4
        x = torch.cat([x, layer1_o], dim=1)
        # B, 512+256, H/4, W/4
        x = self.block3(x)
        # B, 256, H/4, W/4

        x = F.interpolate(
            x,
            size=stem_o.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # B, 256, H/2, W/2
        x = torch.cat([x, stem_o], dim=1)
        # B, 256+64, H/2, W/2
        x = self.block4(x)
        # B, 128, H/2, W/2

        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        # B, 128, H, W
        return self.classifier(x)  # B, num_classes, H, W


class UNetSegmentationModel(nn.Module):
    def __init__(
        self,
        num_classes,
        n_freeze_layers,
    ):
        super().__init__()

        self.encoder = Encoder(n_freeze_layers=n_freeze_layers)

        self.decoder = Decoder(num_classes)

    def forward(self, x):
        output_size = x.shape[-2:]

        layers_o = self.encoder(x)
        logits = self.decoder(
            layers_o,
            output_size=output_size,
        )

        return logits  # B, num_classes, H, W
