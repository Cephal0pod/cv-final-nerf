import torch
import math

def get_rays(H, W, fov, c2w):
    """
    计算每个像素对应的光线原点和方向。
    H, W: 图像高、宽
    fov: 水平视场角 (radian)
    c2w: [4,4] 相机到世界变换矩阵
    返回:
      rays_o: [H*W, 3]
      rays_d: [H*W, 3]
    """
    # 计算焦距（使用 math.tan 处理 float）
    focal = 0.5 * W / math.tan(fov * 0.5)
    # 像素网格坐标
    i, j = torch.meshgrid(
        torch.arange(H, device=c2w.device),
        torch.arange(W, device=c2w.device),
        indexing='ij'
    )
    # 归一化设备坐标系下的方向
    dirs = torch.stack([
        (j + 0.5 - 0.5 * W) / focal,
        -(i + 0.5 - 0.5 * H) / focal,
        -torch.ones_like(i)
    ], dim=-1)  # [H, W, 3]
    # 转换到世界坐标系
    dirs_flat = dirs.reshape(-1, 3)  # [H*W, 3]
    # 旋转
    rays_d = torch.sum(dirs_flat[..., None, :] * c2w[:3, :3], dim=-1)
    # 原点
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def sample_points(origins, dirs, near, far, N_samples):
    """
    对每条光线均匀采样点
    origins, dirs: [N_rays, 3]
    near, far: 近、远裁剪面距离
    N_samples: 每条光线采样点数
    返回:
      pts: [N_rays, N_samples, 3]
      z_vals: [N_rays, N_samples]
    """
    N_rays = origins.shape[0]
    # 在 [near, far] 区间线性采样
    z_vals = torch.linspace(near, far, N_samples, device=origins.device)  # [N_samples]
    z_vals = z_vals.unsqueeze(0).expand(N_rays, N_samples)               # [N_rays, N_samples]
    # 获取采样点坐标
    pts = origins.unsqueeze(1) + dirs.unsqueeze(1) * z_vals.unsqueeze(-1)  # [N_rays, N_samples, 3]
    return pts, z_vals


def volume_rendering(rgb, sigma, z_vals, dirs):
    """
    体渲染核心函数，将体密度和颜色转为像素值
    rgb: [N_rays, N_samples, 3]
    sigma: [N_rays, N_samples, 1]
    z_vals: [N_rays, N_samples]
    dirs: [N_rays, 3]
    返回:
      pixels: [N_rays, 3]
    """
    # 计算相邻深度差
    dists = z_vals[..., 1:] - z_vals[..., :-1]                              # [N_rays, N_samples-1]
    # 增加最后一个远距离
    inf = 1e10 * torch.ones_like(dists[..., :1])                            # [N_rays,1]
    dists = torch.cat([dists, inf], dim=-1)                                  # [N_rays, N_samples]
    # sigma 乘以距离，转为 alpha
    alpha = 1.0 - torch.exp(-sigma[..., 0] * dists)                          # [N_rays, N_samples]
    # 计算权重
    trans = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[:, :-1]                                                                # [N_rays, N_samples]
    weights = alpha * trans                                                   # [N_rays, N_samples]
    # 合成颜色
    pixels = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)                   # [N_rays, 3]
    return pixels


def render(rgb_sigma_fn, H, W, fov, c2w, cfg):
    """
    整体渲染接口
    rgb_sigma_fn: callable, 输入点坐标 [*,3] 输出 (rgb [*,3], sigma [*,1])
    H, W, fov, c2w: 渲染参数
    cfg: 配置字典，包含 data.near, data.far, model.N_samples
    返回:
      image: [H, W, 3]
    """
    # 生成光线
    rays_o, rays_d = get_rays(H, W, fov, c2w)  # [N_rays,3]
    # 采样
    pts, z_vals = sample_points(
        rays_o, rays_d,
        cfg["data"]["near"], cfg["data"]["far"],
        cfg["model"]["N_samples"]
    )
    # 网络前向
    pts_flat = pts.reshape(-1, 3)
    rgb, sigma = rgb_sigma_fn(pts_flat)
    # reshape
    rgb   = rgb.reshape(pts.shape[0], pts.shape[1], 3)
    sigma = sigma.reshape(pts.shape[0], pts.shape[1], 1)
    # 体渲染
    pixels = volume_rendering(rgb, sigma, z_vals, rays_d)  # [N_rays,3]
    # 恢复图像
    return pixels.reshape(H, W, 3)

def sample_pdf(bins, weights, N_samples, det=False):
    # bins: [..., M+1], weights: [..., M] (来自 coarse 阶段的 transmittance*σ)
    # 在 weights 上做归一化，构造 CDF，然后对 M 段做 invert CDF 采样
    pdf = weights + 1e-5
    pdf = pdf / torch.sum(pdf, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [..., M+1]

    # 在 [0,1) 上生成 uniform 样本
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

    # invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds-1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1]-1)
    inds_g = torch.stack([below, above], dim=-1)  # [..., N_samples, 2]

    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(*inds_g.shape[:-1], cdf.shape[-1]),
                         -1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(*inds_g.shape[:-1], bins.shape[-1]),
                          -1, inds_g)

    denom = (cdf_g[...,1] - cdf_g[...,0]).clamp(min=1e-5)
    t = (u - cdf_g[...,0]) / denom
    samples = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])
    return samples

