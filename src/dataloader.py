# src/dataloader.py
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class LLFFDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        """
        data_dir/
          ├── images/         # 渲染图
          ├── transforms.json # 相机内外参
          └── depths.npz      # 可选深度信息
        """
        self.data_dir = data_dir
        meta = json.load(open(os.path.join(data_dir, "transforms.json")))
        self.images = sorted([os.path.join(data_dir,"images",f)
                               for f in os.listdir(os.path.join(data_dir,"images"))])
        self.c2w = [torch.FloatTensor(pose) for pose in meta["poses"]]
        # 根据 split 划分 train/test
        n = len(self.images)
        self.test_idx = set(range(n - int(n * 0.1), n))
        self.idx_list = (list(range(n)) if split=="train"
                         else list(self.test_idx))

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        i = self.idx_list[idx]
        img = Image.open(self.images[i]).convert("RGB")
        img = torch.FloatTensor(np.array(img)/255.0)  # [H,W,3]
        pose = self.c2w[i]                             # [4,4]
        # 生成射线 origin & direction （可在 renderer 中实现）
        return {"image": img, "pose": pose}

def get_dataloader(cfg):
    ds = LLFFDataset(cfg["data"]["data_dir"], split="train")
    return DataLoader(ds,
                      batch_size=cfg["training"]["batch_size"],
                      shuffle=True,
                      num_workers=4)
