from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import rasterio


PRODUCT_IDS = ["6048", "6047", "6042", "6038", "6028"]

L1A_ROOT = Path("data/l1a")
L1C_ROOT = Path("data/l1")
OUT = Path("outputs/l1a_l1c_audit")
OUT.mkdir(parents=True, exist_ok=True)


def find_product(root: Path, level: str, pid: str) -> Path:
    pattern = f"PHISAT-2_{level}_{int(pid):09d}_*"
    candidates = sorted([p for p in root.glob(pattern) if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No {level} product for {pid} under {root}")
    return candidates[0]


def list_files(product: Path) -> dict:
    out = {}
    for p in sorted(product.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(product))] = p.stat().st_size
    return out


def json_keys(product: Path) -> dict:
    out = {}
    for p in sorted(product.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                out[p.name] = sorted(obj.keys())
            else:
                out[p.name] = type(obj).__name__
        except Exception as e:
            out[p.name] = f"ERROR: {type(e).__name__}: {e}"
    return out


def raster_meta(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "rel": str(path),
            "count": src.count,
            "width": src.width,
            "height": src.height,
            "dtypes": ",".join(src.dtypes),
            "crs": str(src.crs),
            "transform": tuple(round(x, 8) for x in src.transform),
            "descriptions": list(src.descriptions),
        }


def read_sample(path: Path, max_side: int = 768) -> np.ndarray:
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        scale = min(1.0, max_side / max(h, w))
        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))
        arr = src.read(
            out_shape=(src.count, out_h, out_w),
            masked=False,
        )
    return arr.astype(np.float32)


def band_stats(path: Path) -> list[dict]:
    arr = read_sample(path)
    rows = []
    for b in range(arr.shape[0]):
        x = arr[b]
        x = x[np.isfinite(x)]
        if x.size == 0:
            rows.append({"band": b, "valid": 0})
            continue
        rows.append({
            "band": b,
            "valid": int(x.size),
            "min": float(np.min(x)),
            "p01": float(np.percentile(x, 1)),
            "p50": float(np.percentile(x, 50)),
            "p99": float(np.percentile(x, 99)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "zero_frac": float(np.mean(x == 0)),
        })
    return rows


def corr_same_shape(path_a: Path, path_c: Path) -> dict:
    a = read_sample(path_a)
    c = read_sample(path_c)

    if a.shape != c.shape:
        return {"same_shape": False, "shape_l1a": str(a.shape), "shape_l1c": str(c.shape)}

    rows = {"same_shape": True, "shape": str(a.shape)}
    n = min(a.shape[0], c.shape[0])
    cors = []
    for b in range(n):
        x = a[b].reshape(-1)
        y = c[b].reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 100 or np.std(x) == 0 or np.std(y) == 0:
            cors.append(float("nan"))
        else:
            cors.append(float(np.corrcoef(x, y)[0, 1]))
    rows["band_corrs"] = cors
    rows["mean_corr"] = float(np.nanmean(cors)) if np.any(np.isfinite(cors)) else float("nan")
    return rows


def main():
    summary_rows = []
    raster_rows = []
    stats_rows = []
    compare_rows = []

    for pid in PRODUCT_IDS:
        print("\n=== product", pid, "===")

        l1a = find_product(L1A_ROOT, "L1A", pid)
        l1c = find_product(L1C_ROOT, "L1", pid)

        files_a = list_files(l1a)
        files_c = list_files(l1c)

        tifs_a = sorted([p for p in l1a.rglob("*.tif*") if p.is_file()])
        tifs_c = sorted([p for p in l1c.rglob("*.tif*") if p.is_file()])

        rel_a = {str(p.relative_to(l1a)): p for p in tifs_a}
        rel_c = {str(p.relative_to(l1c)): p for p in tifs_c}
        common_tifs = sorted(set(rel_a) & set(rel_c))

        summary = {
            "product_id": pid,
            "l1a_path": str(l1a),
            "l1c_path": str(l1c),
            "l1a_n_files": len(files_a),
            "l1c_n_files": len(files_c),
            "l1a_total_mb": round(sum(files_a.values()) / 1e6, 2),
            "l1c_total_mb": round(sum(files_c.values()) / 1e6, 2),
            "l1a_n_tifs": len(tifs_a),
            "l1c_n_tifs": len(tifs_c),
            "common_tifs": len(common_tifs),
            "only_l1a": sorted(set(files_a) - set(files_c)),
            "only_l1c": sorted(set(files_c) - set(files_a)),
            "l1a_json_keys": json_keys(l1a),
            "l1c_json_keys": json_keys(l1c),
        }
        summary_rows.append(summary)

        for level, product, tifs in [("L1A", l1a, tifs_a), ("L1C", l1c, tifs_c)]:
            for tif in tifs:
                meta = raster_meta(tif)
                meta.update({
                    "product_id": pid,
                    "level": level,
                    "relpath": str(tif.relative_to(product)),
                })
                raster_rows.append(meta)

                # Full stats only for multiband/RGB, to keep output manageable.
                if any(k in tif.name for k in ["multiband", "RGB"]):
                    for s in band_stats(tif):
                        s.update({
                            "product_id": pid,
                            "level": level,
                            "relpath": str(tif.relative_to(product)),
                        })
                        stats_rows.append(s)

        for rel in common_tifs:
            if any(k in rel for k in ["multiband", "RGB"]):
                comp = corr_same_shape(rel_a[rel], rel_c[rel])
                comp.update({
                    "product_id": pid,
                    "relpath": rel,
                })
                compare_rows.append(comp)

    # JSON detailed
    (OUT / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    (OUT / "raster_meta.json").write_text(json.dumps(raster_rows, indent=2), encoding="utf-8")
    (OUT / "band_stats.json").write_text(json.dumps(stats_rows, indent=2), encoding="utf-8")
    (OUT / "l1a_l1c_compare.json").write_text(json.dumps(compare_rows, indent=2), encoding="utf-8")

    # CSV compact
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        cols = [
            "product_id",
            "l1a_path",
            "l1c_path",
            "l1a_n_files",
            "l1c_n_files",
            "l1a_total_mb",
            "l1c_total_mb",
            "l1a_n_tifs",
            "l1c_n_tifs",
            "common_tifs",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k) for k in cols})

    with (OUT / "compare.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["product_id", "relpath", "same_shape", "shape", "shape_l1a", "shape_l1c", "mean_corr", "band_corrs"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in compare_rows:
            w.writerow({k: r.get(k) for k in cols})

    print("\nWrote:")
    for p in sorted(OUT.glob("*")):
        print(" ", p)


if __name__ == "__main__":
    main()
