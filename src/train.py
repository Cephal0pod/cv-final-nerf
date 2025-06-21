import os, json, math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from src.utils import load_config
from src.renderer import get_rays, sample_points, volume_rendering
from src.models import get_model
import matplotlib.pyplot as plt

def train():
    cfg    = load_config("C:/Users/20638/Desktop/Study/2024_sem_2/计算机视觉/Final/configs/horns.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type=="cuda":
        torch.backends.cudnn.benchmark = True

    # Model
    model = get_model(cfg).to(device)
    print(f"Using model: {cfg['model']['type']}")
    optim  = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))
    scaler = GradScaler()
    tb     = SummaryWriter(log_dir=cfg.get("log_dir","logs/horns"))

    # Read transforms_train.json
    data_dir = cfg["data"]["data_dir"]
    tf      = os.path.join(data_dir, "transforms_train.json")
    with open(tf,'r') as f:
        meta = json.load(f)

    H = int(meta["h"])
    W = int(meta["w"])
    # 1) 优先用 camera_angle_x
    if "camera_angle_x" in meta:
        fov = float(meta["camera_angle_x"])
    # 2) 否则根据 fl_x 计算水平 FOV
    elif "fl_x" in meta:
        fl = float(meta["fl_x"])    # 焦距 (像素) :contentReference[oaicite:0]{index=0}
        fov = 2 * math.atan((0.5 * W) / fl)
    else:
        raise KeyError("Need camera_angle_x or fl_x in transforms_train.json")

    near = meta.get("near", cfg["data"]["near"])
    far  = meta.get("far",  cfg["data"]["far"])

    # Load frames list
    frames = meta.get("frames")
    assert frames, "transforms_train.json must have 'frames'"

    all_rays, all_tgts = [], []
    for fr in frames:
        # image path under images_4/
        img_path = os.path.join(data_dir, fr["file_path"])
        img = np.array(Image.open(img_path).convert("RGB"), np.float32) / 255.0
        img_t = torch.from_numpy(img).to(device)  # [H,W,3]

        # camera-to-world matrix
        c2w   = torch.FloatTensor(fr["transform_matrix"]).to(device)  # [4,4]

        # generate and flatten rays
        rays_o, rays_d = get_rays(H, W, fov, c2w)   # each [H*W,3]
        all_rays.append((rays_o.view(-1,3), rays_d.view(-1,3)))
        all_tgts.append(img_t.view(-1,3))

    nv = len(all_rays)
    print(f"Loaded {nv} views @ {H}×{W}, FOV={fov:.4f} rad, near={near}, far={far}")

    # Training loop setup
    N_samples   = cfg["model"]["N_samples"]
    N_rand      = cfg["training"].get("N_rand", 1024)
    iters       = cfg["training"]["iters"]
    save_every  = cfg["training"]["save_every"]

    # 4. Prepare history for plotting
    loss_history = []
    psnr_history = []

    model.train()
    for step in range(1, iters+1):
        vid          = torch.randint(0, nv, (1,)).item()
        rays_o_full, rays_d_full = all_rays[vid]
        tgt_full                 = all_tgts[vid]

        # random ray indices
        idxs = torch.randint(0, H*W, (N_rand,), device=device)
        ro   = rays_o_full[idxs]
        rd   = rays_d_full[idxs]
        tgt  = tgt_full[idxs]

        # forward + rendering
        with autocast():
            pts, z_vals = sample_points(ro, rd, near, far, N_samples)
            rgb, sigma  = model(pts.view(-1,3))
            rgb   = rgb.view(N_rand, N_samples, 3)
            sigma = sigma.view(N_rand, N_samples, 1)
            pred  = volume_rendering(rgb, sigma, z_vals, rd)
            loss  = F.mse_loss(pred, tgt)

        # compute loss
        loss_value = loss.item()
        loss_history.append(loss_value)

        # compute psnr as "accuracy"
        psnr_value = -10.0 * math.log10(loss_value + 1e-8)
        psnr_history.append(psnr_value)

        # backward
        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()

        # logging
        if (step + 1) % 100 == 0:
            tb.add_scalar("train/loss", loss.item(), step)
            print(f"step: {step + 1}, loss: {loss.item()}")
        if (step + 1) % save_every == 0:
            ckpt_dir = cfg.get("ckpt_dir","ckpts/horns")
            os.makedirs(ckpt_dir, exist_ok=True)
            p = os.path.join(ckpt_dir, f"ckpt_{step + 1}.pth")
            torch.save(model.state_dict(), p)
            print(f"Saved checkpoint: {p}")

    tb.close()

    # 5. Plot and save curves
    os.makedirs("plots", exist_ok=True)

    # Loss curve
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/train_loss.png")
    plt.close()

    # PSNR curve
    plt.figure()
    plt.plot(psnr_history)
    plt.title("Training PSNR")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/train_psnr.png")
    plt.close()

    print("Saved loss and PSNR plots to ./plots/")

if __name__=="__main__":
    train()

