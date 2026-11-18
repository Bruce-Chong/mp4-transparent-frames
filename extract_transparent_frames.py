"""
Extract MP4 frames as PNGs with transparent backgrounds.

Modes:
  - chroma  : green/blue screen (fast, tunable)
  - rembg   : AI background removal (slower, any background)
  - copy    : write frames as PNG only (no transparency)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _prepare_windows_gpu_dlls() -> None:
    """Let onnxruntime-gpu find CUDA/cuDNN from the toolkit and pip nvidia-* wheels."""
    if sys.platform != "win32":
        return
    import site

    candidates: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    for base in site.getsitepackages():
        nvidia = Path(base) / "nvidia"
        if not nvidia.is_dir():
            continue
        for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            candidates.append(nvidia / sub / "bin")
    seen: set[str] = set()
    for directory in candidates:
        if not directory.is_dir():
            continue
        key = str(directory.resolve())
        if key in seen:
            continue
        seen.add(key)
        os.add_dll_directory(key)
        path = os.environ.get("PATH", "")
        if key not in path.split(os.pathsep):
            os.environ["PATH"] = key + os.pathsep + path


def parse_hex_color(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError("Color must be 6 hex digits, e.g. 00FF00")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def chroma_key_bgr(
    frame_bgr: np.ndarray,
    key_bgr: tuple[int, int, int],
    similarity: float,
    blend: float,
) -> np.ndarray:
    """Return RGBA uint8 array; pixels near key color become transparent."""
    b, g, r = key_bgr
    diff = frame_bgr.astype(np.float32) - np.array([b, g, r], dtype=np.float32)
    dist = np.linalg.norm(diff, axis=2)
    max_dist = (similarity * 255.0) + 1e-6
    alpha = np.clip((dist - max_dist * (1.0 - blend)) / (max_dist * blend + 1e-6), 0.0, 1.0)
    alpha_u8 = (alpha * 255).astype(np.uint8)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return np.dstack([rgb, alpha_u8])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to input .mp4")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <input_stem>_frames)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=("chroma", "rembg", "copy"),
        default="rembg",
        help="Transparency method (default: rembg)",
    )
    parser.add_argument("--fps", type=float, default=None, help="Cap output frame rate (optional)")
    parser.add_argument(
        "--chroma-color",
        type=parse_hex_color,
        default=(0, 255, 0),
        help="RGB hex for chroma key (default: 00FF00 green)",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.35,
        help="Chroma: how wide the color match is (0–1, default 0.35)",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=0.1,
        help="Chroma: edge softness (0–1, default 0.1)",
    )
    parser.add_argument("--prefix", default="frame", help="Output filename prefix")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    out_dir = args.output or (args.input.parent / f"{args.input.stem}_frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print("Could not open video (install codecs or use a supported MP4).", file=sys.stderr)
        return 1

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = 1
    if args.fps and args.fps > 0 and args.fps < src_fps:
        frame_step = max(1, int(round(src_fps / args.fps)))

    rembg_session = None
    if args.mode == "rembg":
        _prepare_windows_gpu_dlls()
        from rembg import new_session, remove as rembg_remove

        rembg_session = new_session()

        def rembg_fn(frame_bgr: np.ndarray) -> np.ndarray:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return np.asarray(rembg_remove(rgb, session=rembg_session))

    key_bgr = (args.chroma_color[2], args.chroma_color[1], args.chroma_color[0])

    index = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if index < args.start:
            index += 1
            continue
        if args.max_frames is not None and written >= args.max_frames:
            break
        if (index - args.start) % frame_step != 0:
            index += 1
            continue

        if args.mode == "copy":
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        elif args.mode == "chroma":
            rgba = chroma_key_bgr(frame, key_bgr, args.similarity, args.blend)
        else:
            rgba = rembg_fn(frame)

        name = f"{args.prefix}_{written + 1:05d}.png"
        Image.fromarray(rgba, mode="RGBA").save(out_dir / name, compress_level=3)
        written += 1
        index += 1
        if written % 10 == 0:
            print(f"Wrote {written} frames...", flush=True)

    cap.release()
    print(f"Done: {written} PNG(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())