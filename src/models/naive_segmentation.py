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

        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)

        # Remove average pooling and classification layers.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # ResNet-50 outputs 2048 feature channels.
        self.out_channels = 2048

        for param in self.backbone[:n_freeze_layers].parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.block1 = ConvBlock(2048, 512)
        self.block2 = ConvBlock(512, 256)
        self.block3 = ConvBlock(256, 128)
        self.block4 = ConvBlock(128, 64)

        self.classifier = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x, output_size):
        # Encoder output is at 1/32 of input resolution.
        x = self.block1(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        x = self.block2(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        x = self.block3(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        x = self.block4(x)

        # Restore the original image resolution.
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return self.classifier(x)


class NaiveSegmentationModel(nn.Module):
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
