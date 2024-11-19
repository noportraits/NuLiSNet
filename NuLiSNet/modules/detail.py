import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from modules.CVMI import New_CVMI_2 as CVMI
from modules.LN import LayerNorm2d as LN
from modules.CSM import CSM
from modules.CSFI import CSFI


class detail(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 16
        self.conv_first = base_conv(3, dim, 3, 1, 1)
        self.down = nn.ModuleList([
            nn.Sequential(
                base_conv(dim, dim, 3, 1, 1),
                nn.Sequential(*[CSM(dim) for _ in range(4)]),
                nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
            ),

            nn.Sequential(
                base_conv(dim * 2, dim * 2, 3, 1, 1),
                nn.Sequential(*[CSM(dim * 2) for _ in range(4)]),
                nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1),
            ),

            nn.Sequential(
                base_conv(dim * 4, dim * 4, 3, 1, 1),
                nn.Sequential(*[CSM(dim * 4) for _ in range(2)]),
                nn.Conv2d(dim * 4, dim * 8, kernel_size=3, stride=2, padding=1),
            )
        ])
        self.middle = nn.Sequential(*[CSM(dim * 8) for _ in range(4)])
        self.pam = nn.ModuleList([
            CVMI(dim),
            CVMI(dim * 2),
            CVMI(dim * 4),
        ])
        self.csfi = CSFI(dim)
        self.up = Decoder(dim)
        self.refine = nn.Sequential(*[CSM(dim) for _ in range(4)])
        self.last = base_conv(dim, 3, 3, 1, 1)

    def forward(self, low1, low2):
        x1 = self.conv_first(low1)
        x2 = self.conv_first(low2)
        x1_1 = self.down[0](x1)
        x2_1 = self.down[0](x2)
        x1_2 = self.down[1](x1_1)
        x2_2 = self.down[1](x2_1)
        x1_3 = self.down[2](x1_2)
        x2_3 = self.down[2](x2_2)
        #############PAM######################
        x1, x2 = self.pam[0](x1, x2)
        x1_1, x2_1 = self.pam[1](x1_1, x2_1)
        x1_2, x2_2 = self.pam[2](x1_2, x2_2)
        x1_3, x2_3 = self.middle(x1_3), self.middle(x2_3)
        [x1, x1_1, x1_2] = self.csfi([x1, x1_1, x1_2])
        [x2, x2_1, x2_2] = self.csfi([x2, x2_1, x2_2])
        x1, x2 = self.up([x1, x1_1, x1_2, x1_3]), self.up([x2, x2_1, x2_2, x2_3])

        return x1, x2


class base_conv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            # LN(input_channel),
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(*[
            nn.ConvTranspose2d(channel * 8, channel * 4, 4, 2, 1),
            nn.GELU(),
        ])
        self.layer2 = nn.Sequential(*[
            nn.Conv2d(channel * 8, channel * 4, 3, 1, 1),
            nn.GELU(),
            nn.Sequential(*[CSM(channel * 4) for _ in range(2)]),
            nn.ConvTranspose2d(channel * 4, channel * 2, 4, 2, 1),
            nn.GELU(),
        ])
        self.layer3 = nn.Sequential(*[
            nn.Conv2d(channel * 4, channel * 2, 3, 1, 1),
            nn.GELU(),
            nn.Sequential(*[CSM(channel * 2) for _ in range(3)]),
            nn.ConvTranspose2d(channel * 2, channel, 4, 2, 1),
            nn.GELU(),
        ])
        self.layer4 = nn.Sequential(*[
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.GELU(),
            nn.Sequential(*[CSM(channel * 1) for _ in range(4)]),
        ])
        self.final_CSM = nn.Sequential(*[CSM(channel) for _ in range(4)])
        self.channel_reduction = nn.Conv2d(channel, 3, 3, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4 = x[3], x[2], x[1], x[0]
        x1 = self.layer1(x1)
        x2 = self.layer2(torch.cat([x2, x1], dim=1))
        x3 = self.layer3(torch.cat([x3, x2], dim=1))
        x4 = self.layer4(torch.cat([x4, x3], dim=1))
        x4 = self.final_CSM(x4)
        x4 = self.channel_reduction(x4)
        return x4
