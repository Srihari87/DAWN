#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path

def find_one(patterns, base, desc):
    for pat in patterns:
        hits = sorted(base.glob(pat))
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"Could not find {desc} in {base}")

def copy_tree(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        if p.is_file():
            out = dst / p.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)

def main():
    parser = argparse.ArgumentParser()

    # REQUIRED by Landseer
    parser.add_argument("--input-dir", default="/data")
    parser.add_argument("--output", default="/output")

    # DAWN mode
    parser.add_argument("--mode", choices=["noisy_verification", "prune"],
                        default="noisy_verification")

    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect DAWN inputs
    cfg = find_one(["*.ini", "configurations/**/*.ini"], in_dir, "config file")
    watermark = find_one(["*watermark*.pkl"], in_dir, "watermark file")
    model = find_one(["*.pt"], in_dir, "model file")

    script = "noisy_verification.py" if args.mode == "noisy_verification" else "prune.py"

    cmd = [
        "python3", script,
        "--config_file", str(cfg),
        "--watermark", str(watermark),
        "--model", str(model),
    ]

    print("[Landseer wrapper] Running:", " ".join(cmd), flush=True)

    proc = subprocess.run(cmd, capture_output=True, text=True)

    (out_dir / "stdout.txt").write_text(proc.stdout)
    (out_dir / "stderr.txt").write_text(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)

    # Copy DAWN outputs
    scores = Path("data/scores")
    if scores.exists():
        copy_tree(scores, out_dir / "data_scores")

    print("[Landseer wrapper] Done")

if __name__ == "__main__":
    main()

