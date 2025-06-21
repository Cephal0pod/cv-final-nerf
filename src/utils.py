# src/utils.py
import yaml
import torch

# src/utils_gauss.py
import os, json
import numpy as np
from PIL import Image

import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def positional_encoding(x, L):
    """
    原版 NeRF 的位置编码：
    γ(p) = [sin(2^i π p), cos(2^i π p)]_{i=0..L-1}
    """
    enc = [x]
    for i in range(L):
        for fn in (torch.sin, torch.cos):
            enc.append(fn((2.0**i) * x))
    return torch.cat(enc, -1)

def load_transforms(split, data_dir):
    path = os.path.join(data_dir, f"transforms_{split}.json")
    with open(path, 'r') as f:
        meta = json.load(f)
    # 从 meta["frames"] 构建 list of (file_path, transform_matrix)
    frames = meta["frames"]
    items = []
    for fr in frames:
        img_fp = os.path.join(data_dir, fr["file_path"])
        mat    = np.array(fr["transform_matrix"], dtype=np.float32).reshape(4,4)
        items.append((img_fp, mat))
    # 也可读取 meta["h"], meta["w"], meta["fl_x"], meta["fl_y"]
    return items, meta

def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr  # (H,W,3)

def compute_fov(meta):
    # 优先 camera_angle_x，否则用 fl_x 计算
    if "camera_angle_x" in meta:
        return float(meta["camera_angle_x"])
    fl = float(meta["fl_x"])
    W  = int(meta["w"])
    import math
    return 2 * math.atan((0.5 * W) / fl)


