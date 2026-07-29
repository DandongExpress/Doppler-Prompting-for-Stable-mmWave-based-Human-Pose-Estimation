"""Preprocess mmRadPose into paired RAD / pose .npy for PULSE training.

mmRadPose provides RAD tensors with OptiTrack mocap ground truth.
This script packs sequences into the unified layout used by ``train.py``.

Usage::

    python tools/preprocess_mmradpose.py --root data/mmRadPose --out data/mmRadPose
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _centre_pelvis(pose: np.ndarray, pelvis_idx: int = 0) -> np.ndarray:
    return pose - pose[..., pelvis_idx : pelvis_idx + 1, :]


def _to_mm(pose: np.ndarray) -> np.ndarray:
    if np.nanmean(np.abs(pose)) < 10.0:
        return pose * 1000.0
    return pose


def pack(rad: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rad.ndim == 3:
        rad = rad[None, ...]
    if pose.ndim == 2:
        pose = pose[None, ...]
    t = min(rad.shape[0], pose.shape[0])
    rad = rad[:t].astype(np.float32)
    pose = _centre_pelvis(_to_mm(pose[:t].astype(np.float32)))
    return rad, pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--split", type=str, default="train")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out or args.root) / args.split
    out.mkdir(parents=True, exist_ok=True)

    # Prefer explicit rad/pose pairs
    rad_files = sorted(root.rglob("*rad*.npy"))
    if not rad_files:
        rad_files = sorted(root.rglob("*.npy"))

    seq_id = 0
    seen = set()
    for rp in rad_files:
        if "pose" in rp.stem.lower() or "joint" in rp.stem.lower():
            continue
        key = str(rp.resolve())
        if key in seen:
            continue
        pose_path = None
        for cand in rp.parent.glob("*.npy"):
            n = cand.stem.lower()
            if cand == rp:
                continue
            if "pose" in n or "joint" in n or "mocap" in n or "gt" in n:
                pose_path = cand
                break
        if pose_path is None:
            continue
        seen.add(key)
        rad, pose = pack(np.load(rp), np.load(pose_path))
        dest = out / f"seq_{seq_id:04d}"
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "rad.npy", rad)
        np.save(dest / "pose.npy", pose)
        print(f"wrote {dest}  rad={rad.shape} pose={pose.shape}")
        seq_id += 1

    if seq_id == 0:
        (out / "README.txt").write_text(
            "No pairs auto-discovered. Manually place:\n"
            "  seq_XXXX/rad.npy  [T,R,A,D]\n"
            "  seq_XXXX/pose.npy [T,J,3]  (mm, pelvis-centred)\n"
        )
        print(f"[preprocess_mmradpose] No pairs found under {root}. Stub written to {out}")
    else:
        print(f"Done. {seq_id} sequences -> {out}")


if __name__ == "__main__":
    main()
