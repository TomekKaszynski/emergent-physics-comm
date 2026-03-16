#!/usr/bin/env python3
"""Extract DINOv2 CLS token features from rendered scene frames.

Supports DINOv2 ViT-S/14 (384-dim) and ViT-L/14 (1024-dim).

Usage:
    python scripts/extract_dinov2.py --dataset kubric/output/collision_dataset --model small
    python scripts/extract_dinov2.py --dataset kubric/output/ramp_dataset --model large --n-frames 8
"""

import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image


MODEL_MAP = {
    'small': ('facebook/dinov2-small', 384),
    'base': ('facebook/dinov2-base', 768),
    'large': ('facebook/dinov2-large', 1024),
}


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 features")
    parser.add_argument("--dataset", required=True, help="Path to scene dataset directory")
    parser.add_argument("--output", default=None, help="Output .pt path")
    parser.add_argument("--model", default="small", choices=MODEL_MAP.keys())
    parser.add_argument("--n-frames", type=int, default=24,
                        help="Number of frames to extract per scene (evenly spaced)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model_name, feat_dim = MODEL_MAP[args.model]

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    from transformers import AutoModel, AutoImageProcessor

    print(f"Loading {model_name}...", flush=True)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    dataset_dir = Path(args.dataset)

    # Load index
    index_path = dataset_dir / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        # Auto-detect scenes
        scene_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("scene_")])
        index = [{'scene_id': i} for i in range(len(scene_dirs))]

    n_scenes = len(index)

    # Determine total frames per scene from first scene
    first_scene = dataset_dir / f"scene_{index[0].get('scene_id', 0):04d}"
    total_frames = len(list(first_scene.glob("rgba_*.png")))
    frame_indices = list(range(0, total_frames, max(1, total_frames // args.n_frames)))[:args.n_frames]

    print(f"Scenes: {n_scenes}, Frames/scene: {args.n_frames} (from {total_frames} total)", flush=True)
    print(f"Feature dim: {feat_dim}", flush=True)

    all_features = torch.zeros(n_scenes, args.n_frames, feat_dim)

    for si in range(n_scenes):
        scene_id = index[si].get('scene_id', si)
        scene_dir = dataset_dir / f"scene_{scene_id:04d}"
        frames = []
        for fi in frame_indices:
            img_path = scene_dir / f"rgba_{fi:05d}.png"
            img = Image.open(img_path).convert("RGB")
            frames.append(img)

        inputs = processor(images=frames, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0, :]
            all_features[si] = cls_tokens.cpu()

        if (si + 1) % 50 == 0:
            print(f"  {si+1}/{n_scenes}", flush=True)
            if device.type == 'mps':
                torch.mps.empty_cache()

    out_path = args.output or str(dataset_dir.parent / f"dinov2_{args.model}_features.pt")
    torch.save({
        'features': all_features,
        'index': index,
        'model': model_name,
        'frames_used': args.n_frames,
        'n_scenes': n_scenes,
    }, out_path)
    print(f"Saved {out_path}: {all_features.shape}", flush=True)


if __name__ == '__main__':
    main()
