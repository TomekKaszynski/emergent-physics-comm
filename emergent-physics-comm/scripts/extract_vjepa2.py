#!/usr/bin/env python3
"""Extract V-JEPA 2 features from rendered scene frames.

V-JEPA 2 produces spatiotemporal features; we mean-pool spatially to get
per-frame 1024-dim vectors.

Usage:
    python scripts/extract_vjepa2.py --dataset kubric/output/collision_dataset --checkpoint path/to/vjepa2.pth
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms


def main():
    parser = argparse.ArgumentParser(description="Extract V-JEPA 2 features")
    parser.add_argument("--dataset", required=True, help="Path to scene dataset directory")
    parser.add_argument("--output", default=None, help="Output .pt path")
    parser.add_argument("--checkpoint", required=True, help="V-JEPA 2 checkpoint path")
    parser.add_argument("--n-frames", type=int, default=24)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load V-JEPA 2 model
    # Note: V-JEPA 2 requires the facebookresearch model hub
    print(f"Loading V-JEPA 2 from {args.checkpoint}...", flush=True)
    model = torch.hub.load("facebookresearch/jepa:main", "vjepa_vitl16_384_16x384x384",
                           pretrained=True)
    model = model.to(device)
    model.eval()

    dataset_dir = Path(args.dataset)
    feat_dim = 1024  # V-JEPA 2 ViT-L

    # Load index
    index_path = dataset_dir / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        scene_dirs = sorted([d for d in dataset_dir.iterdir()
                             if d.is_dir() and d.name.startswith("scene_")])
        index = [{'scene_id': i} for i in range(len(scene_dirs))]

    n_scenes = len(index)

    # Frame preprocessing for V-JEPA 2
    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    first_scene = dataset_dir / f"scene_{index[0].get('scene_id', 0):04d}"
    total_frames = len(list(first_scene.glob("rgba_*.png")))
    frame_indices = list(range(0, total_frames, max(1, total_frames // args.n_frames)))[:args.n_frames]

    print(f"Scenes: {n_scenes}, Frames/scene: {args.n_frames}", flush=True)

    all_features = torch.zeros(n_scenes, args.n_frames, feat_dim)

    for si in range(n_scenes):
        scene_id = index[si].get('scene_id', si)
        scene_dir = dataset_dir / f"scene_{scene_id:04d}"

        frames = []
        for fi in frame_indices:
            img = Image.open(scene_dir / f"rgba_{fi:05d}.png").convert("RGB")
            frames.append(transform(img))

        # Stack into video tensor: (1, C, T, H, W)
        video = torch.stack(frames, dim=1).unsqueeze(0).to(device)

        with torch.no_grad():
            # V-JEPA 2 output: (1, T*H'*W', D) or similar
            # Spatial mean pooling to get per-frame features
            output = model(video)
            # Reshape and pool spatially
            if output.dim() == 3:
                # Assume output is (1, T*spatial, D)
                n_spatial = output.shape[1] // args.n_frames
                output = output.view(1, args.n_frames, n_spatial, feat_dim)
                pooled = output.mean(dim=2)  # (1, T, D)
            else:
                pooled = output
            all_features[si] = pooled.squeeze(0).cpu()

        if (si + 1) % 50 == 0:
            print(f"  {si+1}/{n_scenes}", flush=True)
            if device.type == 'mps':
                torch.mps.empty_cache()

    out_path = args.output or str(dataset_dir.parent / "vjepa2_features.pt")
    torch.save({
        'features': all_features,
        'index': index,
        'model': 'vjepa2_vitl16',
        'frames_used': args.n_frames,
        'n_scenes': n_scenes,
    }, out_path)
    print(f"Saved {out_path}: {all_features.shape}", flush=True)


if __name__ == '__main__':
    main()
