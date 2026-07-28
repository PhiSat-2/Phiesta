from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio


PRODUCT_IDS = ["6048", "6047", "6042", "6038", "6028"]
L1A_ROOT = Path("data/l1a")
L1C_ROOT = Path("data/l1")
OUT = Path("outputs/l1a_l1c_audit_deep")
OUT.mkdir(parents=True, exist_ok=True)


def find_product(root: Path, level: str, pid: str) -> Path:
    candidates = sorted(root.glob(f"PHISAT-2_{level}_{int(pid):09d}_*"))
    candidates = [p for p in candidates if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No {level} product for {pid}")
    return candidates[0]


def read_band(path: Path, band: int, max_side: int = 1024) -> np.ndarray:
    with rasterio.open(path) as src:
        scale = min(1.0, max_side / max(src.height, src.width))
        h = max(1, int(round(src.height * scale)))
        w = max(1, int(round(src.width * scale)))
        arr = src.read(band + 1, out_shape=(h, w)).astype(np.float32)
    return arr


def norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mask = np.isfinite(x)
    if mask.sum() == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(x[mask], [2, 98])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return y.astype(np.float32)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].reshape(-1)
    y = y[mask].reshape(-1)
    if x.size < 100 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def edge_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = norm(x)
    y = norm(y)

    gx1 = np.diff(x, axis=1)
    gx2 = np.diff(y, axis=1)
    gy1 = np.diff(x, axis=0)
    gy2 = np.diff(y, axis=0)

    ex = np.concatenate([gx1.reshape(-1), gy1.reshape(-1)])
    ey = np.concatenate([gx2.reshape(-1), gy2.reshape(-1)])

    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def phase_shift(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    # FFT phase correlation, returns dy, dx in sampled pixels.
    x = norm(x)
    y = norm(y)

    x = x - np.mean(x)
    y = y - np.mean(y)

    fa = np.fft.fft2(x)
    fb = np.fft.fft2(y)
    cps = fa * np.conj(fb)
    cps /= np.abs(cps) + 1e-8

    cc = np.fft.ifft2(cps).real
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    dy, dx = float(peak[0]), float(peak[1])

    h, w = x.shape
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w

    return dy, dx, float(np.sqrt(dy * dy + dx * dx))


def best_orientation_corr(x: np.ndarray, y: np.ndarray) -> dict:
    variants = {
        "same": y,
        "flip_ud": np.flipud(y),
        "flip_lr": np.fliplr(y),
        "rot180": np.flipud(np.fliplr(y)),
    }
    scores = {k: corr(x, v) for k, v in variants.items()}
    best = max(scores, key=lambda k: -999 if np.isnan(scores[k]) else scores[k])
    return {
        "best_orientation": best,
        "best_orientation_corr": scores[best],
        **{f"corr_{k}": v for k, v in scores.items()},
    }


def stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    return {
        "min": float(np.min(x)),
        "p01": float(np.percentile(x, 1)),
        "p50": float(np.percentile(x, 50)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "zero_frac": float(np.mean(x == 0)),
        "sat_frac_65535": float(np.mean(x == 65535)),
    }


def compare_pair(pid: str, label: str, path_a: Path, path_b: Path) -> list[dict]:
    rows = []

    with rasterio.open(path_a) as sa, rasterio.open(path_b) as sb:
        n = min(sa.count, sb.count)

    for b in range(n):
        a = read_band(path_a, b)
        c = read_band(path_b, b)

        row = {
            "product_id": pid,
            "comparison": label,
            "band": b,
            "corr": corr(a, c),
            "edge_corr": edge_corr(a, c),
        }

        dy, dx, mag = phase_shift(a, c)
        row.update({
            "phase_dy_sample_px": dy,
            "phase_dx_sample_px": dx,
            "phase_shift_sample_px": mag,
        })

        row.update(best_orientation_corr(a, c))

        sa = stats(a)
        sc = stats(c)
        for k, v in sa.items():
            row[f"a_{k}"] = v
        for k, v in sc.items():
            row[f"b_{k}"] = v

        rows.append(row)

    return rows


all_rows = []

for pid in PRODUCT_IDS:
    print("product", pid)

    l1a = find_product(L1A_ROOT, "L1A", pid)
    l1c = find_product(L1C_ROOT, "L1", pid)

    l1a_bc = l1a / "bands" / "scene_0_BC_multiband.tiff"
    l1a_rc = l1a / "bands" / "scene_0_RC_multiband.tiff"
    l1c_bc = l1c / "bands" / "scene_0_BC_multiband.tiff"

    all_rows.extend(compare_pair(pid, "L1A_BC_vs_L1C_BC", l1a_bc, l1c_bc))

    if l1a_rc.exists():
        all_rows.extend(compare_pair(pid, "L1A_BC_vs_L1A_RC", l1a_bc, l1a_rc))
        all_rows.extend(compare_pair(pid, "L1A_RC_vs_L1C_BC", l1a_rc, l1c_bc))

json_path = OUT / "deep_metrics.json"
json_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

csv_path = OUT / "deep_metrics.csv"
cols = sorted(set().union(*(r.keys() for r in all_rows)))
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in all_rows:
        w.writerow(r)

print("wrote", csv_path)
print("wrote", json_path)
