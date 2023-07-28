import torch.nn as nn
import torchvision.models as models

from .blocks import *


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        channels = 3

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2**4, width // 2**4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        x = self.model(img)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.resnext = models.resnext50_32x4d(pretrained=True)

        # self.resnext = models.resnext101_32x8d(pretrained=True)
        # self.resnext = resnest50(pretrained=False)

        # self.lrelu = nn.LeakyReLU(inplace=True)

        self.up1 = DenseSumResNetUp(2048, 1024, dropout=0.5)
        self.up2 = DenseSumResNetUp(1024, 512)
        self.up3 = DenseSumResNetUp(512, 256)
        self.up4 = DenseSum1ResNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        output1 = self.resnext.conv1(x)
        output1 = self.resnext.bn1(output1)
        output1 = self.resnext.relu(output1)

        output2 = self.resnext.layer1(output1)
        output3 = self.resnext.layer2(output2)
        output4 = self.resnext.layer3(output3)

        output5 = self.resnext.layer4(output4)

        u1 = self.up1(
            4, output5, output4, output3, output2, output1
        )  # (2048,1024,dropout=0.5)
        u2 = self.up2(3, u1, output3, output2, output1)  # (1024,512)
        u3 = self.up3(2, u2, output2, output1)  # (512,256)
        u4 = self.up4(u3, output1)  # (256,64)
        f = self.final(u4)  # (64,3)

        return f
