import numpy as np
import rasterio
from rasterio.transform import from_origin

from phiesta.l1.l1_event import L1_event


def test_georeference_returns_georeferenced_event(monkeypatch, tmp_path):
    event = L1_event(
        arr=np.zeros((8, 2, 2), dtype=np.float32),
        meta={
            "band_wavelength_nm": [625, 490, 560, 665, 705, 740, 783, 842],
        },
        product_folder="PHISAT-2_L1_000005359_20260512183354_20260512183357_X",
        scene_id=0,
        product_kind="BC",
    )

    captured = {}
    out_path = tmp_path / "out.tif"
    arr = np.arange(8 * 4 * 5, dtype=np.float32).reshape(8, 4, 5)
    transform = from_origin(500000, 5100000, 4.75, 4.75)

    def fake_export(self, output_path=None, **kwargs):
        captured["output_path"] = output_path
        captured["kwargs"] = kwargs

        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=4,
            width=5,
            count=8,
            dtype="float32",
            crs="EPSG:32631",
            transform=transform,
        ) as dst:
            dst.write(arr)
            dst.descriptions = (
                "PAN", "BLUE", "GREEN", "RED",
                "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR",
            )

        return {
            "status": "SUCCESS",
            "path": str(out_path),
            "resolution": 4.75,
            "resampling": "bilinear",
        }

    monkeypatch.setattr(L1_event, "export_georeferenced_tif", fake_export)

    product = event.georeference(
        output_path="custom.tif",
        window_days=10,
        sentinel_backend="download",
        verbose=False,
    )

    assert isinstance(product, L1_event)
    assert product is not event

    assert np.array_equal(product.as_numpy(), arr)
    assert product.meta["georeferenced"] is True
    assert product.meta["path"] == str(out_path)
    assert product.meta["crs"].to_epsg() == 32631
    assert product.meta["transform"] == transform
    assert product.meta["resolution"] == 4.75

    assert product.get_band("RED").shape == (4, 5)
    assert product.get_band("NIR").shape == (4, 5)
    assert product.rgb().shape == (4, 5, 3)

    # Original unchanged
    assert event.as_numpy().shape == (8, 2, 2)

    assert captured == {
        "output_path": "custom.tif",
        "kwargs": {
            "window_days": 10,
            "sentinel_backend": "download",
            "verbose": False,
        },
    }
