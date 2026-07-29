"""Preprocess HuPR into paired RAD / pose .npy for PULSE training.

HuPR already provides range-angle-Doppler tensors. This script:
  1. Loads released RAD heatmaps and 3D joint annotations
  2. Resamples / packs them into the layout expected by ``train.py``
  3. Writes train/val/test splits under ``--out``

Expected raw layout (flexible; adjust KEYS below if your dump differs)::

    <root>/
      <split>/  or sequence folders
        *.npz / *.npy  for RAD   [R, A, D] or [T, R, A, D]
        *.json / *.npy       for poses [J, 3] or [T, J, 3]

Output layout::

    <out>/<split>/seq_XXXX/
        rad.npy   [T, R, A, D]
        pose.npy  [T, J, 3]   (millimetres, pelvis-centred if possible)

Usage::

    python tools/preprocess_hupr.py --root data/HuPR --out data/HuPR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        return np.load(path)  # npz or rename; try npy-compatible
    if path.suffix == ".json":
        return np.asarray(json.loads(path.read_text()), dtype=np.float32)
    raise ValueError(f"Unsupported file type: {path}")


def _centre_pelvis(pose: np.ndarray, pelvis_idx: int = 0) -> np.ndarray:
    """pose: [..., J, 3] -> subtract pelvis joint."""
    return pose - pose[..., pelvis_idx : pelvis_idx + 1, :]


def _to_mm(pose: np.ndarray) -> np.ndarray:
    # Heuristic: if typical magnitude < 10, assume metres
    if np.nanmean(np.abs(pose)) < 10.0:
        return pose * 1000.0
    return pose


def pack_sequence(rad: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rad.ndim == 3:
        rad = rad[None, ...]
    if pose.ndim == 2:
        pose = pose[None, ...]
    t = min(rad.shape[0], pose.shape[0])
    rad = rad[:t].astype(np.float32)
    pose = _centre_pelvis(_to_mm(pose[:t].astype(np.float32)))
    return rad, pose


def discover_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Best-effort pairing of rad/pose files by stem."""
    pairs = []
    rad_cands = list(root.rglob("*rad*.npy")) + list(root.rglob("*RAD*.npy"))
    rad_cands += list(root.rglob("*heatmap*.npy"))
    if not rad_cands:
        rad_cands = list(root.rglob("*.npy"))
    for rp in rad_cands:
        stem = rp.stem.lower().replace("rad", "").replace("heatmap", "")
        for cand in rp.parent.glob("*.npy"):
            if cand == rp:
                continue
            name = cand.stem.lower()
            if "pose" in name or "joint" in name or "skeleton" in name:
                pairs.append((rp, cand))
                break
            if stem and stem in name and ("gt" in name or "label" in name):
                pairs.append((rp, cand))
                break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--split", type=str, default="train")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out or args.root)
    split_out = out / args.split
    split_out.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(root)
    if not pairs:
        print(
            f"[preprocess_hupr] No rad/pose pairs found under {root}.\n"
            "Place files manually as:\n"
            f"  {split_out}/seq_0000/rad.npy  [T,R,A,D]\n"
            f"  {split_out}/seq_0000/pose.npy [T,J,3]"
        )
        # write a tiny placeholder readme for users
        (split_out / "README.txt").write_text(
            "Put seq_*/rad.npy and seq_*/pose.npy here after preprocessing.\n"
        )
        return

    for i, (rp, pp) in enumerate(pairs):
        rad, pose = pack_sequence(_load_array(rp), _load_array(pp))
        dest = split_out / f"seq_{i:04d}"
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "rad.npy", rad)
        np.save(dest / "pose.npy", pose)
        print(f"wrote {dest}  rad={rad.shape} pose={pose.shape}")

    print(f"Done. {len(pairs)} sequences -> {split_out}")


if __name__ == "__main__":
    main()
