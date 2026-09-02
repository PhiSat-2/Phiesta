from __future__ import annotations

import inspect
import sys
import types

import pytest

from phiesta.triplets.full_pipeline import build_full_sentinel_triplet
from phiesta.triplets.proxy_alignment import (
    _estimate_lightglue_homography,
    _load_extractor_and_matcher,
    run_proxy_alignment,
)
from phiesta.triplets.strict_georef import refine_triplet_georeference_strict


class _FakeModule:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.device = None

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self


class _FakeLightGlue(_FakeModule):
    def __init__(self, features=None, **kwargs):
        super().__init__(features=features, **kwargs)
        self.features = features


class _FakeSIFT(_FakeModule):
    pass


def _fake_rbd(value):
    return value


def test_portable_matching_defaults_to_sift():
    assert inspect.signature(_estimate_lightglue_homography).parameters["features"].default == "sift"
    assert inspect.signature(run_proxy_alignment).parameters["features"].default == "sift"
    assert inspect.signature(build_full_sentinel_triplet).parameters["features"].default == "sift"
    assert inspect.signature(refine_triplet_georeference_strict).parameters["features"].default == "sift"


def test_scm_lightglue_namespace_is_used_for_sift(monkeypatch):
    package = types.ModuleType("scm_lightglue")
    package.LightGlue = _FakeLightGlue
    package.SIFT = _FakeSIFT
    utils = types.ModuleType("scm_lightglue.utils")
    utils.rbd = _fake_rbd

    monkeypatch.setitem(sys.modules, "scm_lightglue", package)
    monkeypatch.setitem(sys.modules, "scm_lightglue.utils", utils)

    extractor, matcher, rbd = _load_extractor_and_matcher(
        features="sift",
        max_keypoints=123,
        device="cpu",
    )

    assert isinstance(extractor, _FakeSIFT)
    assert extractor.kwargs["max_num_keypoints"] == 123
    assert extractor.device == "cpu"
    assert isinstance(matcher, _FakeLightGlue)
    assert matcher.features == "sift"
    assert matcher.device == "cpu"
    assert rbd is _fake_rbd


def test_superpoint_has_clear_optional_backend_error(monkeypatch):
    # scm-lightglue intentionally has no SuperPoint extractor. If the original
    # cvg/LightGlue package is unavailable, the error should explain the option.
    monkeypatch.setitem(sys.modules, "lightglue", None)
    monkeypatch.setitem(sys.modules, "lightglue.utils", None)

    with pytest.raises(ImportError, match="features='superpoint'.*original cvg/LightGlue"):
        _load_extractor_and_matcher(
            features="superpoint",
            max_keypoints=123,
            device="cpu",
        )
