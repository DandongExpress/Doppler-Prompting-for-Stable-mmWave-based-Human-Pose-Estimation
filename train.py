"""Training entry for PULSE.

Expects paired RAD tensors and 3D poses under ``data_root`` (see Dataset below).

Example
-------
    python train.py --config configs/pulse_1f_hupr.yaml
    python train.py --config configs/pulse_kf_hupr.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pulse import PULSE, PULSEConfig

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------------
# Dataset: paired RAD (.npy) + pose (.npy)
# ---------------------------------------------------------------------------
# Supported layouts under ``root`` (e.g. data/HuPR or input_data/):
#
#   A) Frame folders
#      root/{train,val,test}/
#          rad/  *.npy   shape [R, A, D]
#          pose/ *.npy   shape [J, 3]   (same stem)
#
#   B) Sequence files
#      root/{train,val,test}/
#          seq_*/rad.npy    [T, R, A, D]
#          seq_*/pose.npy   [T, J, 3]
#
# Multi-frame mode (k_frames > 1) samples contiguous windows ending at t.


class RADPoseDataset(Dataset):
    """Paired mmWave RAD tensors and pelvis-centred 3D joint coordinates."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        k_frames: int = 1,
        pose_scale: float = 1.0,
    ) -> None:
        self.root = Path(root)
        self.split_dir = self.root / split
        if not self.split_dir.is_dir():
            # allow root itself to be the split folder
            self.split_dir = self.root
        self.k_frames = max(1, int(k_frames))
        self.pose_scale = float(pose_scale)

        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self._index: list[tuple[int, int]] = []  # (seq_id, end_frame)
        self._load()

    def _load(self) -> None:
        rad_dir = self.split_dir / "rad"
        pose_dir = self.split_dir / "pose"

        if rad_dir.is_dir() and pose_dir.is_dir():
            # Layout A: per-frame files
            rad_files = sorted(rad_dir.glob("*.npy"))
            rads, poses = [], []
            for rf in rad_files:
                pf = pose_dir / rf.name
                if not pf.exists():
                    continue
                rads.append(np.load(rf).astype(np.float32))
                poses.append(np.load(pf).astype(np.float32) * self.pose_scale)
            if not rads:
                raise FileNotFoundError(
                    f"No paired rad/pose .npy under {rad_dir} and {pose_dir}"
                )
            # treat the whole split as one sequence for windowing
            rad_seq = np.stack(rads, axis=0)
            pose_seq = np.stack(poses, axis=0)
            self.samples.append((rad_seq, pose_seq))
        else:
            # Layout B: sequence subfolders
            seq_dirs = sorted(
                d for d in self.split_dir.iterdir()
                if d.is_dir() and (d / "rad.npy").exists() and (d / "pose.npy").exists()
            )
            if not seq_dirs:
                raise FileNotFoundError(
                    f"No data found under {self.split_dir}. "
                    "Expected rad/+pose/ frame folders or seq_*/rad.npy+pose.npy."
                )
            for d in seq_dirs:
                rad = np.load(d / "rad.npy").astype(np.float32)
                pose = np.load(d / "pose.npy").astype(np.float32) * self.pose_scale
                if rad.ndim != 4 or pose.ndim != 3:
                    raise ValueError(
                        f"{d}: rad must be [T,R,A,D], pose [T,J,3]; "
                        f"got {rad.shape}, {pose.shape}"
                    )
                self.samples.append((rad, pose))

        for sid, (rad, pose) in enumerate(self.samples):
            t = min(rad.shape[0], pose.shape[0])
            start = self.k_frames - 1
            for end in range(start, t):
                self._index.append((sid, end))

        if not self._index:
            raise RuntimeError(
                f"Split too short for k_frames={self.k_frames} under {self.split_dir}"
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        sid, end = self._index[idx]
        rad_seq, pose_seq = self.samples[sid]
        start = end - self.k_frames + 1
        # Per-frame intensity normalisation (Appendix C.5)
        window = rad_seq[start : end + 1].copy()
        for i in range(window.shape[0]):
            m = np.abs(window[i]).mean() + 1e-6
            window[i] = window[i] / m

        pose = pose_seq[end].astype(np.float32)  # predict current frame
        if self.k_frames == 1:
            rad = torch.from_numpy(window[0])           # [R, A, D]
        else:
            rad = torch.from_numpy(window)              # [K, R, A, D]
        return rad, torch.from_numpy(pose.copy())


# ---------------------------------------------------------------------------
# Metrics (shared with eval.py)
# ---------------------------------------------------------------------------

def mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Per Joint Position Error (mm). pred/gt: [..., J, 3]."""
    return torch.norm(pred - gt, dim=-1).mean()


def pa_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Procrustes-aligned MPJPE. pred/gt: [B, J, 3] or [J, 3]."""
    if pred.dim() == 2:
        pred, gt = pred.unsqueeze(0), gt.unsqueeze(0)
    b = pred.shape[0]
    errors = []
    for i in range(b):
        p = pred[i] - pred[i].mean(dim=0, keepdim=True)
        g = gt[i] - gt[i].mean(dim=0, keepdim=True)
        # Kabsch
        h = p.transpose(0, 1) @ g
        u, _, vh = torch.linalg.svd(h)
        r = vh.transpose(0, 1) @ u.transpose(0, 1)
        if torch.det(r) < 0:
            vh = vh.clone()
            vh[-1] *= -1
            r = vh.transpose(0, 1) @ u.transpose(0, 1)
        p_aligned = p @ r
        errors.append(torch.norm(p_aligned - g, dim=-1).mean())
    return torch.stack(errors).mean()


def velocity_metrics(
    pred_seq: torch.Tensor, gt_seq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """MPJVE and AKV over a sequence [T, J, 3] (Delta t = 1 frame)."""
    if pred_seq.shape[0] < 2:
        z = pred_seq.new_tensor(0.0)
        return z, z
    v_hat = pred_seq[1:] - pred_seq[:-1]
    v_gt = gt_seq[1:] - gt_seq[:-1]
    mpjve = torch.norm(v_hat - v_gt, dim=-1).mean()
    akv = torch.norm(v_hat, dim=-1).mean()
    return mpjve, akv


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("Please `pip install pyyaml` to load YAML configs.")
        return yaml.safe_load(text)
    return json.loads(text)


def config_to_pulse(cfg: dict) -> PULSEConfig:
    m = cfg.get("model", cfg)
    return PULSEConfig(
        in_range=int(m.get("in_range", 256)),
        in_angle=int(m.get("in_angle", 128)),
        doppler_bins=int(m.get("doppler_bins", 16)),
        work_range=int(m.get("work_range", 64)),
        work_angle=int(m.get("work_angle", 64)),
        patch_size=int(m.get("patch_size", 4)),
        embed_dim=int(m.get("embed_dim", 32)),
        num_heads=int(m.get("num_heads", 4)),
        neighbor_patches=int(m.get("neighbor_patches", 3)),
        beta=float(m.get("beta", 1.0)),
        depth=int(m.get("depth", 4)),
        mlp_ratio=float(m.get("mlp_ratio", 4.0)),
        dropout=float(m.get("dropout", 0.1)),
        num_joints=int(m.get("num_joints", 17)),
        reg_hidden=int(m.get("reg_hidden", 512)),
        k_frames=int(m.get("k_frames", cfg.get("k_frames", 1))),
    )


def build_loaders(cfg: dict):
    data_cfg = cfg.get("data", {})
    root = data_cfg.get("root", "data/HuPR")
    k = int(cfg.get("model", cfg).get("k_frames", cfg.get("k_frames", 1)))
    pose_scale = float(data_cfg.get("pose_scale", 1.0))
    bs = int(cfg.get("train", {}).get("batch_size", 8))
    nw = int(cfg.get("train", {}).get("num_workers", 0))

    train_set = RADPoseDataset(root, split=data_cfg.get("train_split", "train"),
                               k_frames=k, pose_scale=pose_scale)
    val_split = data_cfg.get("val_split", "val")
    val_root = Path(root) / val_split
    if not val_root.is_dir() and not (Path(root) / "rad").is_dir():
        val_set = train_set
    else:
        try:
            val_set = RADPoseDataset(root, split=val_split, k_frames=k,
                                     pose_scale=pose_scale)
        except FileNotFoundError:
            val_set = train_set

    train_loader = DataLoader(
        train_set, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=bs, shuffle=False, num_workers=nw
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_mpjpe, n = 0.0, 0
    for rad, pose in loader:
        rad, pose = rad.to(device), pose.to(device)
        pred = model(rad)
        total_mpjpe += float(mpjpe(pred, pose)) * rad.size(0)
        n += rad.size(0)
    return {"mpjpe": total_mpjpe / max(n, 1)}


def train(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pulse_cfg = config_to_pulse(cfg)
    model = PULSE(pulse_cfg).to(device)

    train_loader, val_loader = build_loaders(cfg)
    tcfg = cfg.get("train", {})
    lr = float(tcfg.get("lr", 1e-4))
    wd = float(tcfg.get("weight_decay", 0.01))
    epochs = int(tcfg.get("epochs", 100))
    clip = float(tcfg.get("grad_clip", 1.0))
    save_dir = Path(tcfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = tcfg.get("tag", "pulse")

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best = float("inf")

    print(
        f"Device={device} | params={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
        f"| k_frames={pulse_cfg.k_frames} | train={len(train_loader.dataset)}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for rad, pose in train_loader:
            rad, pose = rad.to(device), pose.to(device)
            pred = model(rad)
            loss = PULSE.compute_loss(pred, pose)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim.step()
            running += float(loss) * rad.size(0)

        train_loss = running / max(len(train_loader.dataset), 1)
        val = validate(model, val_loader, device)
        print(
            f"[{epoch:03d}/{epochs}] loss={train_loss:.4f} "
            f"val_mpjpe={val['mpjpe']:.4f} ({time.time()-t0:.1f}s)"
        )

        if val["mpjpe"] < best:
            best = val["mpjpe"]
            ckpt = save_dir / f"{tag}_best.pth"
            torch.save(
                {"model": model.state_dict(), "cfg": pulse_cfg.__dict__, "epoch": epoch},
                ckpt,
            )
            print(f"  saved {ckpt} (best mpjpe={best:.4f})")

    # final checkpoint
    torch.save(
        {"model": model.state_dict(), "cfg": pulse_cfg.__dict__, "epoch": epochs},
        save_dir / f"{tag}_last.pth",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train PULSE")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--data_root", type=str, default=None,
                   help="Override data.root (e.g. input_data/)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.setdefault("data", {})["root"] = args.data_root
    train(cfg)
