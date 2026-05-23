"""
CIFAR 版 Vision Transformer

适配 32x32 输入: patch_size=4 -> 8x8=64 patches + 1 CLS token = 65 tokens
使用 LayerNorm (无 BN running stats 问题, 天然适合 MAML/Meta-SGD)。
配合 torch.func.functional_call 直接被 FedPerMAML / FedPerMetaSGD 使用。

预设:
- vit_tiny:  dim=192, depth=9,  heads=3, mlp=2.0  ~  3M params
- vit_small: dim=256, depth=8,  heads=8, mlp=4.0  ~  6M params
- vit_base:  dim=384, depth=12, heads=6, mlp=4.0  ~ 22M params

注意: ViT 从零在 CIFAR + 小数据 + 联邦设定下显著难于 ResNet, 收敛慢、易过拟合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(attn)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


class ViTCIFAR(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 num_classes=20, embed_dim=256, depth=8,
                 num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, n_patches, dim]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def vit_tiny_cifar(num_classes=20):
    return ViTCIFAR(embed_dim=192, depth=9, num_heads=3, mlp_ratio=2.0,
                    num_classes=num_classes)


def vit_small_cifar(num_classes=20):
    return ViTCIFAR(embed_dim=256, depth=8, num_heads=8, mlp_ratio=4.0,
                    num_classes=num_classes)


def vit_base_cifar(num_classes=20):
    return ViTCIFAR(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                    num_classes=num_classes)


if __name__ == '__main__':
    for name, fn in [('vit_tiny', vit_tiny_cifar),
                     ('vit_small', vit_small_cifar),
                     ('vit_base', vit_base_cifar)]:
        m = fn(num_classes=20)
        n = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 3, 32, 32)
        y = m(x)
        print(f"{name}: params={n/1e6:.2f}M, output={tuple(y.shape)}")
