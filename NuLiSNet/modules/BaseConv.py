from torch import nn as nn
from modules.LN import LayerNorm2d as LN

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
