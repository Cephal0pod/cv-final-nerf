import os
import json
import math
import argparse
import numpy as np
from PIL import Image
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import lpips
except ImportError:
    lpips = None
from tqdm import tqdm
from src.utils import load_config
from src.renderer import get_rays, sample_points, volume_rendering
from src.models import get_model


def render_view(model_fn, H, W, fov, c2w, cfg, chunk_size, device):
    """
    Render a single view in chunks, return [H,W,3] tensor on device
    """
    rays_o, rays_d = get_rays(H, W, fov, c2w)
    N = rays_o.shape[0]
    pixels = torch.zeros((N, 3), device=device)
    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        ro = rays_o[i:j]
        rd = rays_d[i:j]
        pts, z_vals = sample_points(
            ro, rd,
            cfg["data"]["near"], cfg["data"]["far"],
            cfg["model"]["N_samples"]
        )
        rgb, sigma = model_fn(pts.view(-1, 3))
        rgb = rgb.view(j - i, cfg["model"]["N_samples"], 3)
        sigma = sigma.view(j - i, cfg["model"]["N_samples"], 1)
        pixels[i:j] = volume_rendering(rgb, sigma, z_vals, rd)
    return pixels.view(H, W, 3)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained NeRF/TensorRF on test split")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="Which split to evaluate")
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Optional dir to save predicted images")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = get_model(cfg).to(device)
    dir = os.getcwd() + args.checkpoint
    ckpt = torch.load(dir, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # load metadata
    data_dir = cfg["data"]["data_dir"]
    tf = os.path.join(data_dir, f"transforms_{args.split}.json")
    if not os.path.exists(tf):
        raise FileNotFoundError(f"No metadata for split {args.split}: {tf}")
    meta = json.load(open(tf, 'r'))

    H = int(meta.get("h"))
    W = int(meta.get("w"))
    # determine fov
    if "camera_angle_x" in meta:
        fov = float(meta["camera_angle_x"])
    elif "fl_x" in meta:
        fl = float(meta["fl_x"])
        fov = 2 * math.atan((0.5 * W) / fl)
    else:
        raise KeyError("Need camera_angle_x or fl_x in metadata")
    near = meta.get("near", cfg["data"].get("near"))
    far = meta.get("far", cfg["data"].get("far"))

    frames = meta.get("frames") or [{"file_path": f, "transform_matrix": p} for p, f in
                                    zip(meta.get("poses", []), meta.get("file_path", []))]
    psnr_list, ssim_list, lpips_list = [], [], []
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # lpips model
    if lpips:
        lpips_fn = lpips.LPIPS(net='alex').to(device)

    for fr in tqdm(frames, desc="Evaluating"):
        # load GT
        img_path = os.path.join(data_dir, fr["file_path"])
        gt = np.array(Image.open(img_path).convert("RGB"), np.float32) / 255.0
        # render pred
        c2w = torch.FloatTensor(fr["transform_matrix"]).to(device)
        with torch.no_grad():
            pred = render_view(lambda x: model(x), H, W, fov, c2w, cfg, args.chunk_size, device)
        pred_np = pred.clamp(0, 1).cpu().numpy()

        # metrics
        psnr_val = peak_signal_noise_ratio(gt, pred_np, data_range=1.0)
        # 计算最适合的 win_size（一个奇数，且 ≤ 图像最小维度）
        min_dim = min(gt.shape[0], gt.shape[1])
        # 默认想用 7×7 的窗口，但如果图片更小，就用紧挨着的那个最大奇数
        win = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        # 对于 H×W×3 的图像，要指定 channel_axis=2 而不是 multichannel
        ssim_val = structural_similarity(
            gt,
            pred_np,
            data_range=1.0,
            channel_axis=2,
            win_size=win,
        )
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        if lpips:
            # LPIPS expects tensors in [-1,1]
            gt_t = torch.from_numpy((gt * 2 - 1).transpose(2, 0, 1)[None]).to(device)
            pr_t = torch.from_numpy((pred_np * 2 - 1).transpose(2, 0, 1)[None]).to(device)
            lp = lpips_fn(gt_t, pr_t).item()
            lpips_list.append(lp)

        # save pred
        if args.save_dir:
            outp = (pred_np * 255).astype(np.uint8)
            outdir = os.getcwd() + args.save_dir
            Image.fromarray(outp).save(os.path.join(outdir, os.path.basename(fr["file_path"])))

    # summarize
    print(f"PSNR: {np.mean(psnr_list):.4f} ± {np.std(psnr_list):.4f}")
    print(f"SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    if lpips:
        print(f"LPIPS: {np.mean(lpips_list):.4f} ± {np.std(lpips_list):.4f}")


if __name__ == "__main__":
    main()
