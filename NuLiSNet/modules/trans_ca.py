import torch
import torch.nn as nn
from modules.BaseConv import base_conv


class Trans_CA(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super().__init__()
        self.norm = nn.LayerNorm(1)
        self.conv1 = base_conv(dim, dim, 3, 1, 1)
        self.conv2 = base_conv(dim, dim, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        # self.query = nn.Parameter(torch.eyes((1, dim, dim)), requires_grad=True)
        self.query = nn.Parameter(torch.ones((1, dim, dim)), requires_grad=True)
        self.key_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()
        self.dim_linear = nn.ModuleList([nn.Linear(dim, 1) for _ in range(dim)])
        # self.base = nn.Parameter(torch.ones((dim)), requires_grad=False)
        # self.base = 0.5

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.pool(x1)
        x1 = self.conv2(x1)
        x1 = self.pool(x1)
        b, c, h, w = x1.shape
        x1 = x1.permute(0, 2, 3, 1).reshape(b, h * w, c)
        query = self.query.expand(b, -1, -1)
        key = self.key_linear(x1)
        value = self.value_linear(x1)
        attention = self.softmax((query @ key.permute(0, 2, 1)))
        # attention = self.relu(self.softmax((query @ key.permute(0, 2, 1))))
        x1 = (attention @ value)
        _, _, j = x1.shape
        xi = []
        for i in range(j):
            xi.append(self.dim_linear[i](x1[:, i, :]))
        x1 = self.relu(torch.cat(xi, dim=1))
        x1 = x1 + 1
        x1 = x1.unsqueeze(2).unsqueeze(3)
        return x * x1


class Trans_CA2(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super().__init__()
        self.norm = nn.LayerNorm(1)
        self.conv1 = base_conv(dim, dim, 3, 2, 1)
        self.conv2 = base_conv(dim, dim, 3, 2, 1)
        self.conv = base_conv(dim * 2, dim, 1, 1, 0)
        self.query = nn.Parameter(torch.ones((1, dim, dim)), requires_grad=True)
        self.key_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()
        self.dim_linear = nn.ModuleList([nn.Linear(dim, 1) for _ in range(dim)])
        # self.base = nn.Parameter(torch.ones((dim)), requires_grad=False)
        # self.base = 0.5

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        b, c, h, w = x1.shape
        x1 = x1.permute(0, 2, 3, 1).reshape(b, h * w, c)
        query = self.query.expand(b, -1, -1)
        key = self.key_linear(x1)
        value = self.value_linear(x1)
        attention = self.softmax((query @ key.permute(0, 2, 1)))
        # attention = self.relu(self.softmax((query @ key.permute(0, 2, 1))))
        x1 = (attention @ value)
        _, _, j = x1.shape
        xi = []
        for i in range(j):
            xi.append(self.dim_linear[i](x1[:, i, :]))
        x1 = self.relu(torch.cat(xi, dim=1))
        x1 = x1 + 1
        x1 = x1.unsqueeze(2).unsqueeze(3)
        return x * x1


class Trans_CA1(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super().__init__()
        self.norm = nn.LayerNorm(1)
        self.conv1 = base_conv(dim, dim, 3, 1, 1)
        self.conv2 = base_conv(dim, dim, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        # self.query = nn.Parameter(torch.eyes((1, dim, dim)), requires_grad=True)
        self.query = nn.Parameter(torch.ones((1, dim, dim)), requires_grad=True)
        self.key_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()
        self.dim_linear = nn.ModuleList([nn.Linear(dim, 1) for _ in range(dim)])
        # self.base = nn.Parameter(torch.ones((dim)), requires_grad=False)
        # self.base = 0.5

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.pool(x1)
        x1 = self.conv2(x1)
        x1 = self.pool(x1)
        b, c, h, w = x1.shape
        x1 = x1.permute(0, 2, 3, 1).reshape(b, h * w, c)
        query = self.query.expand(b, -1, -1)
        key = self.key_linear(x1)
        value = self.value_linear(x1)
        attention = self.softmax((query @ key.permute(0, 2, 1)))
        # attention = self.relu(self.softmax((query @ key.permute(0, 2, 1))))
        x1 = (attention @ value)
        _, _, j = x1.shape
        xi = []
        for i in range(j):
            xi.append(self.dim_linear[i](x1[:, i, :]))
        x1 = self.relu(torch.cat(xi, dim=1))
        x1 = x1 + 1
        x1 = x1.unsqueeze(2).unsqueeze(3)
        return x * x1


if __name__ == '__main__':
    a = Trans_CA(16)
    x = torch.rand(10, 16, 128, 128)
    x1 = a(x)
    print(x1.shape)
