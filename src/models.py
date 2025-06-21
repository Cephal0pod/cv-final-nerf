import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import positional_encoding


class NeRF(nn.Module):
    """
    基础 NeRF 模型：MLP 体渲染
    """
    def __init__(self, D=8, W=256, L=10):
        super().__init__()
        self.L = L
        in_dim = 3 + 3 * 2 * L
        layers = []
        for i in range(D):
            layers.append(nn.Linear(in_dim if i == 0 else W, W))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)
        self.sigma_out = nn.Linear(W, 1)
        self.rgb_out = nn.Linear(W, 3)

    def forward(self, x):
        # x: [N, 3]
        x_enc = positional_encoding(x, self.L)
        h = self.mlp(x_enc)
        sigma_raw = self.sigma_out(h)
        sigma = F.softplus(sigma_raw)
        rgb = torch.sigmoid(self.rgb_out(h))
        return rgb, sigma


class TensorRF(nn.Module):
    """
    简化版 TensoRF：对 3D 体素网格做 CP 分解并学习显式参数。
    使用分解张量和小型 MLP 输出。
    """
    def __init__(self, resolution=128, rank=8, num_layers=4, encoding_dim=256, pe_L=6):
        super().__init__()
        self.resolution = resolution
        self.rank = rank
        self.pe_L = pe_L
        # 空间分解因子
        self.factor_xy = nn.Parameter(torch.randn(rank, resolution, resolution))
        self.factor_xz = nn.Parameter(torch.randn(rank, resolution, resolution))
        self.factor_yz = nn.Parameter(torch.randn(rank, resolution, resolution))
        # 位置编码维度: 原始坐标(3) + sin/cos 对共 2*self.pe_L*3
        pe_dim = 3 + 3 * 2 * self.pe_L
        in_dim = self.rank + pe_dim
        # MLP 网络
        layers = []
        for i in range(num_layers):
            out_dim = encoding_dim
            layers.append(nn.Linear(in_dim if i == 0 else out_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)
        self.sigma_out = nn.Linear(encoding_dim, 1)
        self.rgb_out = nn.Linear(encoding_dim, 3)

    def forward(self, x):
        # x: [N,3], 假设坐标归一化到 [-1,1]
        # 映射到网格索引范围
        coords = (x + 1) * 0.5 * (self.resolution - 1)
        ix, iy, iz = coords.unbind(-1)
        ix0 = ix.long().clamp(0, self.resolution - 1)
        iy0 = iy.long().clamp(0, self.resolution - 1)
        iz0 = iz.long().clamp(0, self.resolution - 1)
        # 提取 CP 特征 [rank,N]
        f_xy = self.factor_xy[:, iy0, ix0]
        f_xz = self.factor_xz[:, iz0, ix0]
        f_yz = self.factor_yz[:, iz0, iy0]
        feat = (f_xy + f_xz + f_yz).permute(1, 0)  # [N, rank]
        # 位置编码
        pe = positional_encoding(x, self.pe_L)
        inp = torch.cat([feat, pe], dim=-1)
        h = self.mlp(inp)
        sigma = F.softplus(self.sigma_out(h))
        rgb = torch.sigmoid(self.rgb_out(h))
        return rgb, sigma


def get_model(cfg):
    model_type = cfg["model"]["type"].lower()
    if model_type == "tensorrf":
        # TensorRF uses its own positional encoding frequency pe_L (default 6)
        pe_L = cfg["model"].get("pe_L", 6)
        return TensorRF(
            resolution=cfg["model"]["resolution"],
            rank=cfg["model"].get("rank", 8),
            num_layers=cfg["model"].get("num_layers", 4),
            encoding_dim=cfg["model"]["hidden_dim"],
            pe_L=pe_L
        )
    elif model_type == "nerf":
        return NeRF(
            D=cfg["model"]["num_layers"],
            W=cfg["model"]["hidden_dim"],
            L=cfg["model"]["encoding"]["L"]
        )
    else:
        pass


__all__ = ["NeRF", "TensorRF", "get_model"]

