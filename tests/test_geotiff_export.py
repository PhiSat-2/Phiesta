from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from phiesta.triplets.geotiff_export import export_georeferenced_tif


def _write_raster(path, data, *, transform, crs="EPSG:32631", descriptions=None):
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": data.dtype,
        "transform": transform,
        "crs": crs,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if descriptions:
            dst.descriptions = tuple(descriptions)


def test_export_georeferenced_tif_identity_mapping(tmp_path):
    real = tmp_path / "final_triplet" / "phisat2_real_4096.tif"
    real.parent.mkdir(parents=True)
    sentinel = tmp_path / "sentinel_final" / "123_s2b_final_crop_7bands.tif"
    sentinel.parent.mkdir(parents=True)

    arr = np.arange(2 * 8 * 10, dtype=np.uint16).reshape(2, 8, 10)
    transform = from_origin(500000.0, 5100000.0, 5.0, 5.0)
    _write_raster(
        real,
        arr,
        transform=from_origin(1.0, 1.0, 1.0, 1.0),
        descriptions=("PAN", "BLUE"),
    )
    _write_raster(sentinel, arr[:1], transform=transform)

    georef = {
        "method": "sentinel_strict",
        "quality": "good",
        "product_id": "123",
        "H_real_to_s2": np.eye(3).tolist(),
        "H_s2_to_real": np.eye(3).tolist(),
        "paths": {
            "real": str(real),
            "final_sentinel_crop": str(sentinel),
        },
    }

    out = export_georeferenced_tif(
        georef,
        output_path=tmp_path / "out.tif",
        resolution=5.0,
        verbose=False,
    )

    with rasterio.open(out["path"]) as src:
        assert src.crs.to_epsg() == 32631
        assert src.transform.a == 5.0
        assert src.transform.e == -5.0
        assert src.count == 2
        assert src.descriptions == ("PAN", "BLUE")
        # Identity real->Sentinel mapping plus identical 5 m output grid should
        # preserve dimensions and interior pixel values.
        assert src.width == 10
        assert src.height == 8
        got = src.read()
        assert np.array_equal(got, arr)
        assert src.dataset_mask().min() == 255


def test_export_georeferenced_tif_estimates_native_resolution(tmp_path):
    real = tmp_path / "final_triplet" / "phisat2_real_4096.tif"
    real.parent.mkdir(parents=True)
    sentinel = tmp_path / "sentinel_final" / "123_s2b_final_crop_7bands.tif"
    sentinel.parent.mkdir(parents=True)

    arr = np.ones((1, 20, 20), dtype=np.uint16)
    s2_transform = from_origin(600000.0, 5200000.0, 10.0, 10.0)
    _write_raster(real, arr, transform=from_origin(0, 0, 1, 1))
    _write_raster(sentinel, arr, transform=s2_transform)

    # One real pixel corresponds to 0.5 Sentinel pixels -> about 5 m.
    H_real_to_s2 = np.array(
        [
            [0.5, 0.0, 2.0],
            [0.0, 0.5, 3.0],
            [0.0, 0.0, 1.0],
        ]
    )
    georef = {
        "method": "sentinel_strict",
        "quality": "good",
        "product_id": "123",
        "H_real_to_s2": H_real_to_s2.tolist(),
        "paths": {
            "real": str(real),
            "final_sentinel_crop": str(sentinel),
        },
    }

    out = export_georeferenced_tif(
        georef,
        output_path=tmp_path / "native.tif",
        verbose=False,
    )

    assert abs(out["resolution"] - 5.0) < 1e-6
    with rasterio.open(out["path"]) as src:
        assert abs(src.transform.a - 5.0) < 1e-6
        assert abs(src.transform.e + 5.0) < 1e-6
