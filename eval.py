"""Evaluation entry for PULSE.

Reports MPJPE, PA-MPJPE, MPJVE, and AKV without post-hoc smoothing.

Example
-------
    python eval.py --config configs/pulse_1f_hupr.yaml \\
                   --ckpt checkpoints/pulse_1f_hupr.pth

    python eval.py --config configs/pulse_1f_mmradpose.yaml \\
                   --ckpt checkpoints/pulse_1f_hupr.pth --cross_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pulse import PULSE, PULSEConfig
from train import (
    RADPoseDataset,
    config_to_pulse,
    load_config,
    mpjpe,
    pa_mpjpe,
    velocity_metrics,
)


@torch.no_grad()
def evaluate(
    model: PULSE,
    loader: DataLoader,
    device: torch.device,
    sequential: bool = True,
) -> dict:
    """If ``sequential``, group by underlying sequences for MPJVE/AKV."""
    model.eval()
    pos_err, pa_err, n = 0.0, 0.0, 0
    all_mpjve, all_akv = [], []

    # Frame-wise metrics
    preds_cache: list[torch.Tensor] = []
    gts_cache: list[torch.Tensor] = []

    for rad, pose in loader:
        rad, pose = rad.to(device), pose.to(device)
        pred = model(rad)
        b = rad.size(0)
        pos_err += float(mpjpe(pred, pose)) * b
        pa_err += float(pa_mpjpe(pred, pose)) * b
        n += b
        preds_cache.append(pred.cpu())
        gts_cache.append(pose.cpu())

    metrics = {
        "MPJPE": pos_err / max(n, 1),
        "PA-MPJPE": pa_err / max(n, 1),
    }

    # Approximate velocity metrics on the concatenated batch order.
    # For exact sequence-wise MPJVE, prefer layout B (seq folders) and
    # evaluate with shuffle=False over each sequence (below).
    if sequential and hasattr(loader.dataset, "samples"):
        ds: RADPoseDataset = loader.dataset  # type: ignore
        k = ds.k_frames
        for rad_seq, pose_seq in ds.samples:
            t = min(rad_seq.shape[0], pose_seq.shape[0])
            if t < 2:
                continue
            preds = []
            for end in range(k - 1, t):
                start = end - k + 1
                window = rad_seq[start : end + 1].copy()
                for i in range(window.shape[0]):
                    m = np.abs(window[i]).mean() + 1e-6
                    window[i] = window[i] / m
                if k == 1:
                    x = torch.from_numpy(window[0]).unsqueeze(0).to(device)
                else:
                    x = torch.from_numpy(window).unsqueeze(0).to(device)
                preds.append(model(x).squeeze(0).cpu())
            pred_seq = torch.stack(preds, dim=0)
            gt_seq = torch.from_numpy(pose_seq[k - 1 : t].astype(np.float32))
            v_e, a_e = velocity_metrics(pred_seq, gt_seq)
            all_mpjve.append(float(v_e))
            all_akv.append(float(a_e))
    else:
        pred_cat = torch.cat(preds_cache, dim=0)
        gt_cat = torch.cat(gts_cache, dim=0)
        v_e, a_e = velocity_metrics(pred_cat, gt_cat)
        all_mpjve.append(float(v_e))
        all_akv.append(float(a_e))

    metrics["MPJVE"] = float(np.mean(all_mpjve)) if all_mpjve else 0.0
    metrics["AKV"] = float(np.mean(all_akv)) if all_akv else 0.0
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PULSE")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument(
        "--cross_dataset",
        action="store_true",
        help="Use config data.root for testing with a foreign checkpoint.",
    )
    p.add_argument("--split", type=str, default=None, help="Override test split name")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.setdefault("data", {})["root"] = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    if "cfg" in ckpt:
        pulse_cfg = PULSEConfig(**{k: v for k, v in ckpt["cfg"].items()
                                   if k in PULSEConfig.__dataclass_fields__})
    else:
        pulse_cfg = config_to_pulse(cfg)

    model = PULSE(pulse_cfg).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    data_cfg = cfg.get("data", {})
    root = data_cfg.get("root", "data/HuPR")
    split = args.split or data_cfg.get("test_split", data_cfg.get("val_split", "test"))
    k = pulse_cfg.k_frames
    pose_scale = float(data_cfg.get("pose_scale", 1.0))

    try:
        dataset = RADPoseDataset(root, split=split, k_frames=k, pose_scale=pose_scale)
    except FileNotFoundError:
        dataset = RADPoseDataset(root, split="val", k_frames=k, pose_scale=pose_scale)

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    metrics = evaluate(model, loader, device)

    print("=" * 48)
    print(f"Checkpoint : {args.ckpt}")
    print(f"Data root  : {root}  split={split}  cross={args.cross_dataset}")
    for k_, v_ in metrics.items():
        print(f"  {k_:<10}: {v_:.4f}")
    print("=" * 48)


if __name__ == "__main__":
    main()
