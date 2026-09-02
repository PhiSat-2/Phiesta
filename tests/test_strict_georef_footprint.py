from __future__ import annotations

import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from phiesta.triplets.strict_georef import georef_from_strict_result


def _write(path, *, width=2, height=2, transform=None, crs="EPSG:4326"):
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = transform or from_origin(10.0, 20.0, 0.01, 0.01)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        transform=transform,
        crs=crs,
    ) as dst:
        dst.write(np.ones((1, height, width), dtype=np.float32))


def test_georef_footprint_uses_outer_edges_without_half_pixel_shift(tmp_path):
    final_dir = tmp_path / "final_triplet"
    real = final_dir / "phisat2_real_4096.tif"
    sentinel_warped = final_dir / "sentinel_final_warped_to_real_4096.tif"
    simulated_warped = final_dir / "simulated_final_warped_to_real_4096.tif"
    metadata_path = final_dir / "final_triplet_metadata.json"
    final_sentinel = tmp_path / "sentinel_final" / "123_s2b_final_crop_7bands.tif"

    transform = from_origin(10.0, 20.0, 0.01, 0.01)
    for path in (real, sentinel_warped, simulated_warped, final_sentinel):
        _write(path, transform=transform)

    metadata_path.write_text(
        json.dumps(
            {
                "product_id": "123",
                "real_path": str(real),
                "sentinel_warped_path": str(sentinel_warped),
                "simulated_warped_path": str(simulated_warped),
            }
        ),
        encoding="utf-8",
    )

    triplet = {
        "paths": {
            "real": str(real),
            "sentinel": str(sentinel_warped),
            "simulated": str(simulated_warped),
            "metadata": str(metadata_path),
            "final_sentinel_crop": str(final_sentinel),
        }
    }
    strict = {
        "report": {
            "product_id": "123",
            "source": "simulated",
            "features": "superpoint",
            "H_s2_to_real_strict": np.eye(3).tolist(),
            "H_residual_source_to_real": np.eye(3).tolist(),
            "metrics": {
                "inliers": 250,
                "inlier_ratio": 0.8,
                "error_median_px": 1.0,
                "error_p90_px": 2.0,
                "error_p95_px": 2.5,
            },
            "paths": {"real": str(real)},
        }
    }

    georef = georef_from_strict_result(triplet, strict)

    expected = np.array(
        [
            [10.0, 20.0],
            [10.02, 20.0],
            [10.02, 19.98],
            [10.0, 19.98],
        ]
    )
    assert np.allclose(np.asarray(georef["corners_lonlat"]), expected, atol=1e-10)
    assert np.allclose(georef["center_lonlat"], [10.01, 19.99], atol=1e-10)
