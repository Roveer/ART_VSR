import argparse
import glob
import os
import re
from pathlib import Path

import torch
from PIL import Image

from artvsr.checkpoint import DEFAULT_CHECKPOINT
from artvsr.pipeline import build_artvsr_pipeline, run_artvsr_video


PROJECT_ROOT = Path(__file__).resolve().parent


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)]


def load_frames(folder_path):
    if not folder_path or not os.path.exists(folder_path):
        return []
    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    files.sort(key=lambda p: natural_sort_key(os.path.basename(p)))
    return files


def run_input_dir(args, pipe):
    input_files = load_frames(args.input_dir)
    if args.max_frames > 0:
        input_files = input_files[:args.max_frames]
    if not input_files:
        raise FileNotFoundError(f"No image frames found in {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    input_frames = []
    for path in input_files:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            input_frames.append(image.resize((width * 4, height * 4), Image.BICUBIC))
    output_frames = run_artvsr_video(
        pipe,
        input_frames,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        color_fix=not args.no_color_fix,
    )
    if len(output_frames) != len(input_files):
        print(f"[Warning] output frames {len(output_frames)} != input files {len(input_files)}")
    for path, image in zip(input_files, output_frames):
        image.save(os.path.join(args.output_dir, os.path.basename(path)))
    print(f"[Inference] Results saved to: {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="ART-VSR image-sequence inference")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "results" / "demo"))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=17)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--no_color_fix", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print("[Inference] Loading ART-VSR pipeline...")
    pipe = build_artvsr_pipeline(checkpoint=args.checkpoint, device=args.device, torch_dtype=torch.bfloat16, enable_vram=True)
    run_input_dir(args, pipe)


if __name__ == "__main__":
    main()
