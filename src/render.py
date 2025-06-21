import os
import json
import argparse
import math
import torch
import imageio
from tqdm import tqdm
from src.utils import load_config
from src.renderer import get_rays, sample_points, volume_rendering
from src.models import get_model


def render_view(model_fn, H, W, fov, c2w, cfg, chunk_size):
    """
    分块渲染单个视角
    """
    rays_o, rays_d = get_rays(H, W, fov, c2w)
    N_rays = rays_o.shape[0]
    pixels = torch.zeros((N_rays, 3), device=rays_o.device)
    for i in range(0, N_rays, chunk_size):
        end = min(i + chunk_size, N_rays)
        ro = rays_o[i:end]
        rd = rays_d[i:end]
        pts, z_vals = sample_points(
            ro, rd,
            cfg["data"]["near"], cfg["data"]["far"],
            cfg["model"]["N_samples"]
        )
        rgb, sigma = model_fn(pts.reshape(-1, 3))
        rgb = rgb.view(end - i, cfg["model"]["N_samples"], 3)
        sigma = sigma.view(end - i, cfg["model"]["N_samples"], 1)
        pixels[i:end] = volume_rendering(rgb, sigma, z_vals, rd)
    return pixels.reshape(H, W, 3)


def main():
    parser = argparse.ArgumentParser(description="Render novel views from trained checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint .pth file")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for renders and video")
    parser.add_argument("--chunk_size", type=int, default=32768, help="Rays per chunk")
    parser.add_argument("--save_video", action="store_true", help="Compile frames into video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for video")
    parser.add_argument("--dup", type=int, default=1, help="Frame duplication factor for video")
    # 可选：插值生成时长，以秒为单位
    parser.add_argument("--duration", type=float, default=None, help="Total video duration in seconds for interpolation")
    args = parser.parse_args()

    # Load config and model
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    # Load checkpoint
    dir = os.getcwd() + args.checkpoint
    ckpt = torch.load(dir, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # Load metadata
    data_dir = cfg["data"]["data_dir"]
    tf_path = os.path.join(data_dir, "transforms_test.json")
    if not os.path.exists(tf_path):
        tf_path = os.path.join(data_dir, "transforms.json")
    with open(tf_path, 'r') as f:
        meta = json.load(f)

    H = int(meta["h"])
    W = int(meta["w"])
    # Determine fov
    if "camera_angle_x" in meta:
        fov = float(meta["camera_angle_x"])
    elif "fl_x" in meta:
        fl = float(meta["fl_x"])
        fov = 2 * math.atan((0.5 * W) / fl)
    else:
        raise KeyError("transforms.json must contain 'camera_angle_x' or 'fl_x'")

    near = meta.get("near", cfg["data"].get("near", 0.1))
    far  = meta.get("far",  cfg["data"].get("far", 5.0))

    # Extract poses
    frames = meta.get("frames", None)
    if frames is None:
        poses = meta.get("poses")
        assert poses, "No poses or frames found in metadata"
    else:
        poses = [f["transform_matrix"] for f in frames]

    # 插值关键帧至固定时长
    if args.duration is not None:
        total_frames = int(args.duration * args.fps)
        key_n = len(poses)
        gaps = key_n - 1
        if total_frames < key_n:
            raise ValueError(f"Duration*FPS={total_frames} less than key frames {key_n}")
        per_gap = (total_frames - key_n) // gaps
        extra = (total_frames - key_n) % gaps
        interp = []
        import torch as _torch
        for i in range(gaps):
            p0 = _torch.tensor(poses[i], dtype=_torch.float32)
            p1 = _torch.tensor(poses[i+1], dtype=_torch.float32)
            interp.append(p0.numpy())
            n_i = per_gap + (1 if i < extra else 0)
            for j in range(1, n_i+1):
                alpha = j / (n_i + 1)
                mat = p0 * (1 - alpha) + p1 * alpha
                interp.append(mat.numpy())
        interp.append(_torch.tensor(poses[-1], dtype=_torch.float32).numpy())
        poses = interp
        print(f"Interpolated to {len(poses)} frames for duration {args.duration}s at {args.fps}fps")

    os.makedirs(args.out_dir, exist_ok=True)
    frame_paths = []

    # Render loop
    with torch.no_grad():
        for idx, pose in enumerate(tqdm(poses, desc="Rendering views")):
            c2w = torch.FloatTensor(pose).to(device)
            img = render_view(lambda pts: model(pts), H, W, fov, c2w, cfg, args.chunk_size)
            img_np = (img.clamp(0,1).cpu().numpy() * 255).astype('uint8')
            path = os.path.join(args.out_dir, f"frame_{idx:03d}.png")
            imageio.imwrite(path, img_np)
            frame_paths.append(path)

    # Save video with possible frame duplication
    if args.save_video:
        video_path = os.getcwd() + os.path.join(args.out_dir, 'video.mp4')
        with imageio.get_writer(video_path, fps=args.fps, format='ffmpeg') as writer:
            for fp in frame_paths:
                frame = imageio.imread(fp)
                for _ in range(args.dup):
                    writer.append_data(frame)
        print(f"Saved video to {video_path}")

    print(f"Rendered {len(frame_paths)} frames to {args.out_dir}")

if __name__ == "__main__":
    main()


