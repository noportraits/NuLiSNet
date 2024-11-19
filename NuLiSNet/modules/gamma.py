import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import os


class query_Attention1(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Parameter(torch.ones((1, 3, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        x1 = x
        k = self.k(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q.expand(B, -1, -1).view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x1 = (attn @ v).transpose(1, 2).reshape(B, 3, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)
        # print(x.shape)
        return x1


class query_Attention2(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = 1
        head_dim = dim // 1
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Parameter(torch.ones((1, 9 * 9, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        x1 = x
        k = self.k(x1).reshape(B, N, C).permute(0, 2, 1)
        v = self.v(x1).reshape(B, N, C)
        q = self.q.expand(B, -1, -1).view(B, -1, C)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x1 = (attn @ v)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)
        # print(x.shape)
        return x1


class query_SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pos_embed2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn1 = query_Attention1(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x1 = x + self.pos_embed1(x)
        x1 = x1.flatten(2).transpose(1, 2)
        x1 = self.norm1(x1)
        x1 = self.attn1(x1)

        x1 = self.drop_path(x1)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x1 = x1 + self.drop_path(self.norm2(x1))
        return x1


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_embedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            # nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(out_channels // 2),
            # nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class Global_pred(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_heads=1, type='exp'):
        super(Global_pred, self).__init__()
        if type == 'exp':
            self.alpha_base = nn.Parameter(torch.ones((1)), requires_grad=False)  # False in exposure correction
            self.beta_base = nn.Parameter(torch.ones((1)), requires_grad=False)  # False in exposure correction
            self.gamma_base = nn.Parameter(torch.ones((1)), requires_grad=False)  # False in exposure correction
            self.g_base = nn.Parameter(torch.eye((9)), requires_grad=True)
        else:
            self.gamma_base = nn.Parameter(torch.ones((1)), requires_grad=True)
        self.color_base = nn.Parameter(torch.eye((3)), requires_grad=True)  # basic color matrix
        # main blocks
        self.conv = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        self.conv_large = conv_embedding(in_channels, out_channels)
        self.generator = query_SABlock(dim=out_channels, num_heads=num_heads)
        self.alpha_linear = nn.Linear(out_channels, 1, bias=True)
        self.beta_linear = nn.Linear(out_channels, 1, bias=True)
        self.gamma_linear = nn.Linear(out_channels, 1, bias=True)
        self.g_linear = nn.Linear(out_channels, 1, bias=True)
        # self.color_linear = nn.Linear(out_channels, 1)

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            if name == 'generator.attn.v.weight':
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, _, _, _ = x.shape
        x, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv(x)
        x = self.conv_large(x)
        x1 = self.generator(x)

        alpha = x1[:, 0].unsqueeze(1)
        alpha = self.alpha_linear(alpha).squeeze(-1) + self.alpha_base
        beta = x1[:, 1].unsqueeze(1)
        beta = self.beta_linear(beta).squeeze(-1) + self.beta_base
        gamma = x1[:, 2].unsqueeze(1)
        gamma = self.gamma_linear(gamma).squeeze(-1) + self.gamma_base
        return alpha, beta, gamma


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # net = Local_pred_new().cuda()
    img = torch.Tensor(10, 16, 128, 128)
    global_net = Global_pred()
    alpha, beta, gamma = global_net(img)
    # print(beta, alpha, gamma)
