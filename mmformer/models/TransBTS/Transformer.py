import torch
import torch.nn as nn
from models.TransBTS.IntmdSequential import IntermediateSequential


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)


    def forward(self, x):
        return self.net(x)

class mmTransformerModel(nn.Module):
    def __init__(
        self,
        modal_num,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        self.self_attention_list = []
        self.self_ffn_list = []
        for i in range(modal_num):
            self.self_attention_list.append(list())
            self.self_ffn_list.append(list())
        self.compress_list = []
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.expand_list = []
        self.depth = depth
        self.modal_num = modal_num
        for j in range(self.depth):
            # for i in range(modal_num):
            #     self.self_attention_list[i].append(
            #         Residual(
            #             PreNormDrop(
            #                 dim,
            #                 dropout_rate,
            #                 SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
            #             )
            #         )
            #     )
            #     self.self_ffn_list[i].append(
            #         Residual(
            #             PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
            #         )
            #     )
            # self.compress_list.append(nn.Linear(dim, int(dim/self.modal_num)))
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        # int(dim/self.modal_num),
                        dim,
                        dropout_rate,
                        # SelfAttention(int(dim/self.modal_num), heads=heads, dropout_rate=attn_dropout_rate),
                        SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    # PreNorm(int(dim/self.modal_num), FeedForward(int(dim/self.modal_num), mlp_dim, dropout_rate))
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                )
            )
            # self.expand_list.append(nn.Linear(int(dim/self.modal_num), dim))

        for i in range(modal_num):
            self.self_attention_list[i] = nn.ModuleList(self.self_attention_list[i])
            self.self_ffn_list[i] = nn.ModuleList(self.self_ffn_list[i])
        # self.self_attention_list = nn.ModuleList(self.self_attention_list)
        # self.self_ffn_list = nn.ModuleList(self.self_ffn_list)
        # self.compress_list = nn.ModuleList(self.compress_list)
        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)
        # self.expand_list = nn.ModuleList(self.expand_list)

    def forward(self, x):
        for j in range(self.depth):
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x