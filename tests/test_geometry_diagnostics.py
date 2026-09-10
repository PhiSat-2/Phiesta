import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phiesta import (
    edge_overlay,
    interband_shift_table,
    local_interband_shift_field,
    register_bands,
)


class FakeEvent:
    BAND_NAMES = {
        "BLUE": 0,
        "GREEN": 1,
        "RED": 2,
        "NIR": 3,
    }

    def __init__(self, arr, meta=None):
        self._arr = np.asarray(arr)
        self.meta = dict(meta or {"product_id": "5359"})
        self.product_folder = "PHISAT-2_L1_000005359_fake"

    def get_band(self, band):
        if isinstance(band, str):
            band = self.BAND_NAMES[band.upper()]
        return self._arr[int(band)]

    def to_cube(self, bands="all", band_axis=0, copy=False):
        return self._arr.copy() if copy else self._arr

    def as_numpy(self):
        return self._arr

    def get_meta(self):
        return self.meta

    def with_array(self, arr, meta_updates=None):
        meta = dict(self.meta)
        meta.update(meta_updates or {})
        return FakeEvent(arr, meta=meta)

    def _resolve_band_index_for_registration(self, band):
        if isinstance(band, str):
            return self.BAND_NAMES[band.upper()]
        return int(band)


def test_interband_shift_table_has_target_to_master_sign_convention():
    rng = np.random.default_rng(0)
    master = rng.normal(size=(128, 128)).astype(np.float32)
    target = np.roll(master, shift=(3, -5), axis=(0, 1))

    event = FakeEvent(np.stack([master, target]))

    table = interband_shift_table(
        event,
        master_band=0,
        target_bands=[1],
        max_side=128,
    )

    row = table.iloc[0]
    assert row["status"] == "ok"
    assert row["dy_px"] == pytest.approx(-3, abs=0.01)
    assert row["dx_px"] == pytest.approx(5, abs=0.01)
    assert row["corr_after"] > row["corr_before"]


def test_local_interband_shift_field_recovers_tilewise_translation():
    rng = np.random.default_rng(1)
    tiles = []
    shifted = []

    for _ in range(4):
        tile = rng.normal(size=(64, 64)).astype(np.float32)
        tiles.append(tile)
        shifted.append(np.roll(tile, shift=(2, -3), axis=(0, 1)))

    master = np.block([[tiles[0], tiles[1]], [tiles[2], tiles[3]]])
    target = np.block([[shifted[0], shifted[1]], [shifted[2], shifted[3]]])

    event = FakeEvent(np.stack([master, target]))

    field = local_interband_shift_field(
        event,
        master_band=0,
        target_band=1,
        window_size=64,
        stride=64,
        max_shifts=(10, 10),
    )

    ok = field[field["status"] == "ok"]
    assert len(ok) == 4
    assert float(ok["dy_px"].median()) == pytest.approx(-2, abs=0.01)
    assert float(ok["dx_px"].median()) == pytest.approx(3, abs=0.01)


def test_edge_overlay_shape_and_range():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(64, 64)).astype(np.float32)
    b = np.roll(a, shift=(1, 2), axis=(0, 1))
    event = FakeEvent(np.stack([a, b]))

    overlay = edge_overlay(event, band_a=0, band_b=1)

    assert overlay.shape == (64, 64, 3)
    assert overlay.dtype == np.float32
    assert float(overlay.min()) >= 0
    assert float(overlay.max()) <= 1


def test_register_bands_returns_aligned_copy():
    rng = np.random.default_rng(3)
    master = rng.normal(size=(96, 96)).astype(np.float32)
    target = np.roll(master, shift=(2, -4), axis=(0, 1))
    event = FakeEvent(np.stack([master, target, master, target]))

    aligned = register_bands(event, master_band=0, max_shifts=(10, 10))

    corr = np.corrcoef(
        aligned.get_band(0)[10:-10, 10:-10].ravel(),
        aligned.get_band(1)[10:-10, 10:-10].ravel(),
    )[0, 1]

    assert aligned is not event
    assert corr > 0.98
    assert aligned.meta["band_registration_info"]["master_band"] == 0


def test_interband_shift_table_string_master_excludes_master():
    rng = np.random.default_rng(11)
    red = rng.normal(size=(96, 96)).astype(np.float32)
    nir = np.roll(red, shift=(1, -2), axis=(0, 1))
    event = FakeEvent(np.stack([red, red, red, nir]))

    table = interband_shift_table(
        event,
        master_band="RED",
        target_bands="all",
        max_side=96,
    )

    assert int(table["master_band_index"].iloc[0]) == 2
    assert 2 not in set(table["target_band"].astype(int))
    assert 3 in set(table["target_band"].astype(int))
