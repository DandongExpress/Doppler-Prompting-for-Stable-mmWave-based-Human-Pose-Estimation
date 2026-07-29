"""Preprocess XRF55: reconstruct RAD from RA + RD maps (paper Appendix C).

XRF55 releases 2D maps. We form a unified RAD tensor::

    w[r,a] = M_RA[r,a] / (sum_a' M_RA[r,a'] + eps)     (Eq. 15)
    H[r,a,d] = w[r,a] * M_RD[r,d]                      (Eq. 16)

Output layout matches ``train.py`` (seq folders with rad.npy / pose.npy).

Usage::

    python tools/preprocess_xrf55.py --root data/XRF55 --out data/XRF55
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def reconstruct_rad(ra: np.ndarray, rd: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Args:
        ra: [R, A] or [T, R, A]
        rd: [R, D] or [T, R, D]
    Returns:
        H: [R, A, D] or [T, R, A, D]
    """
    single = ra.ndim == 2
    if single:
        ra = ra[None, ...]
        rd = rd[None, ...]
    t, r, a = ra.shape
    d = rd.shape[-1]
    assert rd.shape[0] == t and rd.shape[1] == r

    w = ra / (ra.sum(axis=-1, keepdims=True) + eps)          # [T, R, A]
    # H[t,r,a,d] = w[t,r,a] * rd[t,r,d]
    h = w[..., None] * rd[..., None, :]                      # [T, R, A, D]
    return h[0] if single else h.astype(np.float32)


def _centre_pelvis(pose: np.ndarray, pelvis_idx: int = 0) -> np.ndarray:
    return pose - pose[..., pelvis_idx : pelvis_idx + 1, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out or args.root) / args.split
    out.mkdir(parents=True, exist_ok=True)

    # Best-effort discovery: folders containing ra.npy + rd.npy (+ pose.npy)
    seq_id = 0
    candidates = sorted(root.rglob("ra.npy")) + sorted(root.rglob("RA.npy"))
    if not candidates:
        print(
            f"[preprocess_xrf55] No ra.npy found under {root}.\n"
            "For each sequence provide ra.npy [T,R,A], rd.npy [T,R,D], pose.npy [T,J,3].\n"
            f"Writing stub folder under {out}"
        )
        (out / "README.txt").write_text(
            "Place seq_*/ with ra.npy, rd.npy, pose.npy then re-run this script,\n"
            "or directly write reconstructed rad.npy + pose.npy for train.py.\n"
        )
        return

    for ra_path in candidates:
        parent = ra_path.parent
        rd_path = parent / ("rd.npy" if (parent / "rd.npy").exists() else "RD.npy")
        pose_path = None
        for name in ("pose.npy", "joints.npy", "gt.npy"):
            if (parent / name).exists():
                pose_path = parent / name
                break
        if not rd_path.exists() or pose_path is None:
            continue

        ra = np.load(ra_path).astype(np.float32)
        rd = np.load(rd_path).astype(np.float32)
        pose = np.load(pose_path).astype(np.float32)
        rad = reconstruct_rad(ra, rd, eps=args.eps)
        if pose.ndim == 2:
            pose = pose[None, ...]
        if rad.ndim == 3:
            rad = rad[None, ...]
        t = min(rad.shape[0], pose.shape[0])
        rad, pose = rad[:t], _centre_pelvis(pose[:t])
        if np.nanmean(np.abs(pose)) < 10.0:
            pose = pose * 1000.0

        dest = out / f"seq_{seq_id:04d}"
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "rad.npy", rad.astype(np.float32))
        np.save(dest / "pose.npy", pose.astype(np.float32))
        print(f"wrote {dest}  rad={rad.shape} pose={pose.shape}")
        seq_id += 1

    print(f"Done. {seq_id} sequences -> {out}")


if __name__ == "__main__":
    main()
