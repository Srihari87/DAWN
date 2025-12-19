#!/usr/bin/env python3
import argparse
import io
import os
import pickle
from pathlib import Path

import numpy as np
import torch


def _pickle_load_force_cpu(path: str):
    """
    DAWN score files are often `pickle.dump()` of objects that contain torch tensors.
    When unpickling, torch storages are restored via torch.load(BytesIO(...)) with no map_location.
    We monkeypatch torch.load ONLY for BytesIO/byte-buffer loads so those storages map to CPU.
    """
    orig_torch_load = torch.load

    def patched_torch_load(f, *args, **kwargs):
        # Only patch the INTERNAL loads during unpickling (BytesIO / buffered IO).
        if isinstance(f, (io.BytesIO, io.BufferedReader, io.BufferedIOBase)):
            kwargs.setdefault("map_location", torch.device("cpu"))
        return orig_torch_load(f, *args, **kwargs)

    torch.load = patched_torch_load
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    finally:
        torch.load = orig_torch_load


def _as_numpy_2d(x):
    """Convert tensors/arrays of many shapes into a 2D numpy array [N, D]."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        arr = x.numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        return None

    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    # collapse everything except first dim
    return arr.reshape(arr.shape[0], -1)


def _collect_arrays(obj, out):
    """Recursively collect tensors/ndarrays from nested structures."""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        out.append(obj)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_arrays(v, out)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_arrays(v, out)
        return


def extract_best_2d(obj):
    """
    Find the 'best' candidate logits-like array:
      - prefer 2D or collapsible arrays
      - prefer those with second dim = 10 (MNIST/CIFAR classes) if available
      - otherwise choose the largest (N*D)
    """
    found = []
    _collect_arrays(obj, found)

    candidates = []
    for item in found:
        arr2d = _as_numpy_2d(item)
        if arr2d is None:
            continue
        candidates.append(arr2d)

    if not candidates:
        return None

    # Prefer [N,10] if exists
    ten_class = [c for c in candidates if c.ndim == 2 and c.shape[1] == 10]
    if ten_class:
        ten_class.sort(key=lambda a: a.shape[0] * a.shape[1], reverse=True)
        return ten_class[0]

    candidates.sort(key=lambda a: a.shape[0] * a.shape[1], reverse=True)
    return candidates[0]


def run_dawn_like_score(ground_truth_path: str, watermark_path: str):
    """
    Minimal "score" that won't crash:
    - loads two DAWN outputs
    - extracts a 2D array from each
    - returns simple distance-based statistic
    """
    gt_obj = _pickle_load_force_cpu(ground_truth_path)
    wm_obj = _pickle_load_force_cpu(watermark_path)

    gt = extract_best_2d(gt_obj)
    wm = extract_best_2d(wm_obj)

    if gt is None or wm is None:
        raise RuntimeError(
            "Could not extract any tensor/ndarray data from the provided pkl files.\n"
            f"ground_truth: {ground_truth_path}\nwatermark: {watermark_path}"
        )

    # Make shapes compatible by matching feature dim
    d = min(gt.shape[1], wm.shape[1])
    gt = gt[:, :d]
    wm = wm[:, :d]

    # Compare watermark logits vs ground-truth distribution
    gt_mean = gt.mean(axis=0)
    wm_mean = wm.mean(axis=0)

    l2 = float(np.linalg.norm(wm_mean - gt_mean))
    cos = float(
        1.0 - (np.dot(wm_mean, gt_mean) / (np.linalg.norm(wm_mean) * np.linalg.norm(gt_mean) + 1e-12))
    )

    return {
        "gt_shape": list(gt.shape),
        "wm_shape": list(wm.shape),
        "l2_mean_distance": l2,
        "cosine_distance": cos,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", required=True, help="*_ground_truth_*.pkl from DAWN")
    ap.add_argument("--watermark", required=True, help="*_watermark_*.pkl from DAWN")
    ap.add_argument("--output", required=True, help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    result = run_dawn_like_score(args.ground_truth, args.watermark)

    # Write both JSON-ish and CSV for convenience
    (outdir / "dawn_wrapper_results.txt").write_text(str(result) + "\n")

    # Small CSV
    csv_path = outdir / "dawn_wrapper_results.csv"
    keys = ["gt_shape", "wm_shape", "l2_mean_distance", "cosine_distance"]
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        f.write(",".join([f"\"{result[k]}\"" if isinstance(result[k], list) else str(result[k]) for k in keys]) + "\n")

    print("[wrapper] wrote:", str(csv_path))
    print("[wrapper] wrote:", str(outdir / "dawn_wrapper_results.txt"))


if __name__ == "__main__":
    main()
