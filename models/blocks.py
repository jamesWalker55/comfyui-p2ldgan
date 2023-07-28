import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class DenseSumResNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(DenseSumResNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.conv1 = nn.Conv2d(512, 1024, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 512, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(64, 256, 4, 2, 1, bias=False)

        self.upsample = nn.Upsample(scale_factor=2)

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        n,
        p,
        skip_input1=None,
        skip_input2=None,
        skip_input3=None,
        skip_input4=None,
    ):
        x = self.model(p)
        if n == 1:
            skip_input1 = self.upsample(skip_input1)
            x = torch.add(x, skip_input1)
        elif n == 2:
            skip_input2 = self.conv4(skip_input2)
            skip_input2 = self.upsample(skip_input2)
            x = torch.add(x, skip_input2)
            x = torch.add(x, skip_input1)
        elif n == 3:
            skip_input2 = self.conv2(skip_input2)
            skip_input3 = self.conv3(skip_input3)
            x = torch.add(x, skip_input3)
            x = torch.add(x, skip_input2)
            x = torch.add(x, skip_input1)
        elif n == 4:
            skip_input2 = self.conv1(skip_input2)

            skip_input3 = self.conv2(skip_input3)
            skip_input3 = self.conv1(skip_input3)

            skip_input4 = self.conv3(skip_input4)
            skip_input4 = self.conv1(skip_input4)

            x = torch.add(x, skip_input4)
            x = torch.add(x, skip_input3)
            x = torch.add(x, skip_input2)
            x = torch.add(x, skip_input1)

        return x


def amplify_img(imgs):
    return nn.functional.interpolate(
        imgs, torch.Size([imgs.shape[-2] * 2, imgs.shape[-1] * 2]), mode="nearest"
    )


class DenseSum1ResNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(DenseSum1ResNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.add(x, skip_input)
        return x
