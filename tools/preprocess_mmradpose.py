"""Preprocess mmRadPose cubes into the RAD tensors consumed by PULSE.

The Zenodo release stores one complex cube per recording::

    data_cube_parsed_<id>.npz
    shape [T, azimuth, elevation, range, chirps] = [T, 4, 4, 64, 128]

Axis ``range`` is already the unwindowed range spectrum (the range FFT of the
fast-time ADC samples). This script does not run that FFT again. It finishes
the standard FMCW chain used for HuPR and mmRadPose (Appendix C) and the
range–angle resample in Appendix D.3:

    Doppler FFT along chirps
    angle FFT across the azimuth x elevation virtual array
    magnitude
    sum over elevation, leaving a single azimuth axis
    16 evenly spaced Doppler bins
    bilinear resize of the range–angle plane to 64 x 64

Static clutter is not removed: nothing is subtracted along chirps before the
Doppler FFT. There is no window, no CFAR, no Doppler crop, and no pooling
or averaging of Doppler bins. The 16 bins are a stride through the
fftshifted 128-bin spectrum (``128 // 16``), which keeps both signs of
velocity and the zero-Doppler bin.

Skeletons are converted to millimetres when they are stored in metres and
are centred on the root joint (index 0). Recordings are written into
subject-disjoint ``train`` / ``val`` / ``test`` folders.

Usage::

    python tools/preprocess_mmradpose.py --root data/mmRadPose --out data/mmRadPose
    python tools/preprocess_mmradpose.py --self_check
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# Subject-disjoint split for the public release: p3 validates, p1/p2/p12
# are held out, and the remaining participants train.
VAL_SUBJECTS = {3}
TEST_SUBJECTS = {1, 2, 12}

_ID_RE = re.compile(r"p(\d+)_an(\d+)_ac(\d+)_r(\d+)", re.IGNORECASE)


def even_indices(n: int, k: int) -> np.ndarray:
    """k strictly increasing indices spread evenly across ``0 .. n-1``."""
    if k <= 0 or n < k:
        raise ValueError(f"cannot take {k} bins from length {n}")
    if n % k == 0:
        return np.arange(0, n, n // k, dtype=np.int64)
    idx = np.clip(np.round(np.linspace(0, n, k, endpoint=False)), 0, n - 1)
    idx = idx.astype(np.int64)
    if len(np.unique(idx)) != k:
        raise ValueError(f"cannot take {k} distinct bins from length {n}")
    return idx


def canonicalize_cube(cube: np.ndarray) -> np.ndarray:
    """Return ``[T, azimuth, elevation, range, chirps]``.

    The released cube is already in that order. A permutation is accepted
    when the four non-time axes are ``{4, 4, 64, 128}``; the two length-4
    axes keep their relative order (azimuth, then elevation).
    """
    if cube.ndim != 5:
        raise ValueError(f"expected a 5D radar cube, got shape {cube.shape}")
    dims = cube.shape[1:]
    if dims == (4, 4, 64, 128):
        return cube
    idx_128 = [i for i, n in enumerate(dims) if n == 128]
    idx_64 = [i for i, n in enumerate(dims) if n == 64]
    idx_4 = [i for i, n in enumerate(dims) if n == 4]
    if len(idx_128) == 1 and len(idx_64) == 1 and len(idx_4) == 2:
        order = (0, idx_4[0] + 1, idx_4[1] + 1, idx_64[0] + 1, idx_128[0] + 1)
        return np.transpose(cube, order)
    raise ValueError(
        f"unrecognised cube shape {cube.shape}; expected [T, 4, 4, 64, 128] "
        "(frame, azimuth, elevation, range, chirps)"
    )


def bilinear_ra(rad: np.ndarray, out_range: int, out_angle: int) -> np.ndarray:
    """Bilinear resize of the range–angle plane. ``rad`` is ``[T, R, A, D]``.

    Matches ``F.interpolate(..., mode="bilinear", align_corners=False)``
    used when the model resamples to the working grid (Appendix D.3).
    """
    t, r, a, d = rad.shape
    if (r, a) == (out_range, out_angle):
        return np.ascontiguousarray(rad, dtype=np.float32)
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.ascontiguousarray(rad, dtype=np.float32))
    x = x.permute(0, 3, 1, 2).reshape(t * d, 1, r, a)
    with torch.no_grad():
        y = F.interpolate(
            x, size=(out_range, out_angle), mode="bilinear", align_corners=False
        )
    y = y.reshape(t, d, out_range, out_angle).permute(0, 2, 3, 1).contiguous()
    return y.numpy()


def frames_to_rad(
    cube: np.ndarray,
    *,
    doppler_bins: int = 16,
    range_bins: int = 64,
    angle_bins: int = 64,
    range_is_fft: bool = True,
) -> np.ndarray:
    """Complex cube ``[T, 4, 4, 64, 128]`` -> magnitude RAD ``[T, 64, 64, 16]``.

    ``range_is_fft`` is true for ``data_cube_parsed_*.npz``: the range axis is
    already the range spectrum. Set it false only for raw fast-time ADC samples.
    """
    cube = canonicalize_cube(np.asarray(cube))
    if not np.iscomplexobj(cube):
        if not getattr(frames_to_rad, "_warned_real", False):
            print(
                "warning: cube is real-valued; Doppler is read from the "
                "phase across chirps, which a magnitude-only cube does not keep"
            )
            frames_to_rad._warned_real = True
        cube = cube.astype(np.complex64)
    else:
        cube = cube.astype(np.complex64, copy=False)

    if not range_is_fft:
        # Raw fast-time samples. No window and no mean subtraction.
        cube = np.fft.fft(cube, axis=-2)

    # Doppler FFT along chirps. The chirp mean is left in place, so the
    # zero-Doppler bin still carries static returns.
    spec = np.fft.fftshift(np.fft.fft(cube, axis=-1), axes=-1)

    # Angle FFT on the stored azimuth x elevation virtual array, then one
    # azimuth axis by summing elevation magnitudes.
    spec = np.fft.fft(spec, axis=1)
    spec = np.fft.fft(spec, axis=2)
    spec = np.fft.fftshift(spec, axes=(1, 2))
    mag = np.abs(spec).sum(axis=2, dtype=np.float32)  # [T, azimuth, range, Doppler]
    mag = np.transpose(mag, (0, 2, 1, 3))  # [T, range, azimuth, Doppler]

    mag = mag[..., even_indices(mag.shape[-1], doppler_bins)]
    return bilinear_ra(mag, range_bins, angle_bins)


def load_cube(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    with np.load(path) as blob:
        if "real" in blob.files and "imag" in blob.files:
            cube = blob["real"] + 1j * blob["imag"]
        else:
            cube = blob[blob.files[0]]
        return np.array(cube)


def recording_key(path: Path) -> str:
    stem = path.stem
    prefix = "data_cube_parsed_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return stem


def split_for_key(key: str) -> str:
    match = _ID_RE.search(key)
    if match is None:
        return "train"
    subject = int(match.group(1))
    if subject in TEST_SUBJECTS:
        return "test"
    if subject in VAL_SUBJECTS:
        return "val"
    return "train"


def find_skeleton(cube_path: Path, key: str) -> Path | None:
    name = f"skeleton_{key}.npy"
    seen: set[Path] = set()
    for base in (cube_path.parent, *cube_path.parents):
        if base in seen:
            continue
        seen.add(base)
        for candidate in (
            base / "skeletons" / name,
            base / "skeleton" / name,
            base / name,
        ):
            if candidate.is_file():
                return candidate
        if base.parent == base:
            break
    return None


def prepare_pose(pose: np.ndarray, pelvis_idx: int) -> np.ndarray:
    """``[T, J, 3]`` in millimetres, centred on the root joint."""
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim == 2 and pose.shape[-1] != 3 and pose.shape[-1] % 3 == 0:
        pose = pose.reshape(pose.shape[0], -1, 3)
    elif pose.ndim == 2 and pose.shape[-1] == 3:
        pose = pose[None, ...]
    elif pose.ndim == 3 and pose.shape[-1] != 3 and pose.shape[1] == 3:
        pose = np.transpose(pose, (0, 2, 1))
    if pose.ndim != 3 or pose.shape[-1] != 3:
        raise ValueError(f"expected pose [T, J, 3], got {pose.shape}")
    if np.nanmean(np.abs(pose)) < 10.0:
        pose = pose * 1000.0
    if not 0 <= pelvis_idx < pose.shape[1]:
        raise ValueError(
            f"pelvis index {pelvis_idx} is outside J={pose.shape[1]}"
        )
    return pose - pose[:, pelvis_idx : pelvis_idx + 1, :]


def discover_cubes(root: Path) -> list[Path]:
    cubes = sorted(root.rglob("data_cube_parsed_*.npz"))
    cubes += sorted(root.rglob("data_cube_parsed_*.npy"))
    # Stable unique paths.
    seen = set()
    unique = []
    for path in cubes:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def preprocess_root(
    root: Path,
    out: Path,
    *,
    doppler_bins: int,
    range_bins: int,
    angle_bins: int,
    range_is_fft: bool,
    pelvis_idx: int,
) -> int:
    cubes = discover_cubes(root)
    if not cubes:
        print(
            f"No data_cube_parsed_* files under {root}.\n"
            "Expected the Zenodo raw release, for example:\n"
            "  <root>/radar/data_cube_parsed_p1_an0_ac0_r0.npz\n"
            "  <root>/skeletons/skeleton_p1_an0_ac0_r0.npy"
        )
        return 0

    written = {"train": 0, "val": 0, "test": 0}
    for cube_path in cubes:
        key = recording_key(cube_path)
        skeleton_path = find_skeleton(cube_path, key)
        if skeleton_path is None:
            print(f"skip {cube_path.name}: no skeleton_{key}.npy")
            continue
        cube = load_cube(cube_path)
        rad = frames_to_rad(
            cube,
            doppler_bins=doppler_bins,
            range_bins=range_bins,
            angle_bins=angle_bins,
            range_is_fft=range_is_fft,
        )
        pose = prepare_pose(np.load(skeleton_path), pelvis_idx)
        if pose.shape[1] != 17 and not getattr(preprocess_root, "_noted_joints", False):
            print(
                f"pose has {pose.shape[1]} joints; set model.num_joints "
                "in the mmRadPose config to this count"
            )
            preprocess_root._noted_joints = True
        t = min(rad.shape[0], pose.shape[0])
        if t == 0:
            print(f"skip {key}: empty cube or skeleton")
            continue
        rad, pose = rad[:t], pose[:t]
        split = split_for_key(key)
        dest = out / split / key
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "rad.npy", rad)
        np.save(dest / "pose.npy", pose)
        written[split] += 1
        print(
            f"wrote {dest}  rad={tuple(rad.shape)} pose={tuple(pose.shape)}"
        )

    total = sum(written.values())
    print(
        f"Done. {total} recordings -> {out} "
        f"(train={written['train']}, val={written['val']}, test={written['test']})"
    )
    return total


def run_self_check() -> None:
    """Shape, bin selection, and the two operations this chain does not do."""
    idx = even_indices(128, 16)
    assert np.array_equal(idx, np.arange(0, 128, 8))
    assert 64 in idx  # zero-Doppler bin after fftshift

    cube = np.zeros((2, 4, 4, 64, 128), dtype=np.complex64)
    cube[:, :, :, 10, :] = 1
    rad = frames_to_rad(cube)
    assert rad.shape == (2, 64, 64, 16), rad.shape
    assert rad.dtype == np.float32
    assert np.isfinite(rad).all()

    ra = rad[0].sum(axis=-1)
    assert int(ra.sum(axis=1).argmax()) == 10
    assert int(ra.sum(axis=0).argmax()) > 24
    doppler = rad[0].sum(axis=(0, 1))
    assert int(doppler.argmax()) == 8
    assert float(doppler.sum()) > 0

    # A tone one bin off the stride grid must not leak in through pooling.
    n = np.arange(128, dtype=np.float64)
    tone = np.exp(2j * np.pi * n / 128).astype(np.complex64)
    off = np.zeros_like(cube[:1])
    off[:, :, :, 10, :] = tone
    off_rad = frames_to_rad(off)
    assert float(off_rad.sum()) < 1e-3 * float(rad[0].sum())

    swapped = np.transpose(cube, (0, 4, 1, 3, 2))  # [T, chirps, az, range, el]
    assert swapped.shape == (2, 128, 4, 64, 4)
    swapped_rad = frames_to_rad(swapped)
    assert np.allclose(swapped_rad, rad, rtol=1e-4, atol=1e-4)

    assert split_for_key("p1_an0_ac1_r0") == "test"
    assert split_for_key("p12_an2_ac9_r1") == "test"
    assert split_for_key("p3_an1_ac2_r0") == "val"
    assert split_for_key("p4_an0_ac1_r0") == "train"
    print("self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--doppler_bins", type=int, default=16)
    parser.add_argument("--range_bins", type=int, default=64)
    parser.add_argument("--angle_bins", type=int, default=64)
    parser.add_argument(
        "--input",
        choices=("range_fft", "adc"),
        default="range_fft",
        help="range_fft: parsed Zenodo cubes (default). adc: raw fast-time samples.",
    )
    parser.add_argument("--pelvis_idx", type=int, default=0)
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return
    if not args.root:
        parser.error("--root is required unless --self_check is set")

    root = Path(args.root)
    out = Path(args.out) if args.out else root
    n = preprocess_root(
        root,
        out,
        doppler_bins=args.doppler_bins,
        range_bins=args.range_bins,
        angle_bins=args.angle_bins,
        range_is_fft=args.input == "range_fft",
        pelvis_idx=args.pelvis_idx,
    )
    if n == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
