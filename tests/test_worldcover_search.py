import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import phiesta.remote.worldcover as wc


def test_worldcover_class_aliases():
    assert wc.resolve_worldcover_class("mangrove") == (95, "mangroves")
    assert wc.resolve_worldcover_class("mangroves") == (95, "mangroves")
    assert wc.resolve_worldcover_class(95) == (95, "mangroves")
    assert wc.resolve_worldcover_class("built-up") == (50, "built_up")


def test_count_class_in_local_tile(tmp_path):
    path = tmp_path / "tile.tif"
    arr = np.full((20, 20), 10, dtype=np.uint8)
    arr[5:10, 5:10] = 95

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=20,
        height=20,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0.0, 2.0, 0.1, 0.1),
        nodata=0,
    ) as dst:
        dst.write(arr, 1)

    valid, mangrove = wc._count_class_in_local_tile(
        path,
        box(0.0, 0.0, 2.0, 2.0),
        class_code=95,
    )
    assert valid == 400
    assert mangrove == 25


def test_search_defaults(monkeypatch, tmp_path):
    features = [
        {
            "id": "a",
            "properties": {"productIdentifier": "PHISAT-2_L1_000000001_X"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        },
        {
            "id": "b",
            "properties": {"productIdentifier": "PHISAT-2_L1_000000002_X"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        },
    ]

    class FakeClient:
        cache_dir = tmp_path

        def iter_ref_data(self, **kwargs):
            yield from features

    def fake_stats(feature, worldcover, **kwargs):
        pid = "1" if feature["id"] == "a" else "2"
        frac = 1e-6 if pid == "1" else 5e-7
        return {
            "product_id": pid,
            "product_identifier": feature["properties"]["productIdentifier"],
            "filename": None,
            "start_datetime": None,
            "center_lon": 0.5,
            "center_lat": 0.5,
            "worldcover_class": "mangroves",
            "worldcover_code": 95,
            "worldcover_fraction": frac,
            "worldcover_pixels": 1,
            "worldcover_valid_pixels": 1_000_000,
            "spatial_tolerance_km": kwargs["spatial_tolerance_km"],
            "worldcover_tiles": "",
        }

    monkeypatch.setattr(wc, "worldcover_stats_for_feature", fake_stats)

    out = wc.search_l1_worldcover(FakeClient(), "mangrove", verbose=False)

    assert out["product_id"].tolist() == ["1"]
    assert out.iloc[0]["worldcover_fraction"] == 1e-6
    assert out.iloc[0]["spatial_tolerance_km"] == 30.0




def test_categorical_count_from_stats():
    stats = {
        "valid_pixels": 1000,
        "histogram": [[17, 983], [95, 10]],
    }
    valid, target = wc._categorical_count_from_stats(stats, 95)
    assert valid == 1000
    assert target == 17


def test_search_keeps_service_failures_as_uncertain_candidates(monkeypatch):
    features = [
        {
            "id": "ok",
            "properties": {"productIdentifier": "PHISAT-2_L1_000000001_X"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        },
        {
            "id": "fail",
            "properties": {"productIdentifier": "PHISAT-2_L1_000000002_X"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        },
    ]

    class FakeClient:
        def iter_ref_data(self, **kwargs):
            yield from features

    def fake_stats(feature, worldcover, **kwargs):
        if feature["id"] == "fail":
            raise RuntimeError("504 Gateway Time-out")
        return {
            "product_id": "1",
            "product_identifier": feature["properties"]["productIdentifier"],
            "filename": None,
            "start_datetime": None,
            "center_lon": 0.5,
            "center_lat": 0.5,
            "worldcover_class": "mangroves",
            "worldcover_code": 95,
            "worldcover_fraction": 0.01,
            "worldcover_pixels": 10,
            "worldcover_valid_pixels": 1000,
            "spatial_tolerance_km": kwargs["spatial_tolerance_km"],
            "worldcover_items": "",
        }

    monkeypatch.setattr(wc, "worldcover_stats_for_feature", fake_stats)

    out = wc.search_l1_worldcover(
        FakeClient(),
        "mangrove",
        verbose=False,
    )

    assert set(out["product_id"].astype(str)) == {"1", "2"}
    ok = out[out["product_id"].astype(str) == "1"].iloc[0]
    failed = out[out["product_id"].astype(str) == "2"].iloc[0]
    assert ok["worldcover_status"] == "matched"
    assert failed["worldcover_status"] == "uncertain"
    assert "504 Gateway Time-out" in failed["worldcover_error"]


def test_search_can_still_fail_fast(monkeypatch):
    feature = {
        "id": "fail",
        "properties": {"productIdentifier": "PHISAT-2_L1_000000002_X"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        },
    }

    class FakeClient:
        def iter_ref_data(self, **kwargs):
            yield feature

    def fake_stats(*args, **kwargs):
        raise RuntimeError("504 Gateway Time-out")

    monkeypatch.setattr(wc, "worldcover_stats_for_feature", fake_stats)

    import pytest
    with pytest.raises(RuntimeError, match="504 Gateway Time-out"):
        wc.search_l1_worldcover(
            FakeClient(),
            "mangrove",
            include_uncertain=False,
            verbose=False,
        )
