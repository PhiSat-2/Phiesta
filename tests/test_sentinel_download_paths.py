from __future__ import annotations

import zipfile
from pathlib import Path

from phiesta.triplets.sentinel_download import (
    _find_extracted_safe_dir,
    download_cdse_product_zip,
)


PRODUCT_NAME = "S2B_MSIL1C_20260605T180909_N0512_R084_T12TVL_20260605T214135.SAFE"


def _mark_complete_safe(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "MTD_MSIL1C.xml").write_text("<xml />", encoding="utf-8")
    return path


def test_find_extracted_safe_dir_ignores_partial_directory(tmp_path):
    products_dir = tmp_path / "products"
    partial = products_dir / PRODUCT_NAME
    partial.mkdir(parents=True)

    assert _find_extracted_safe_dir(products_dir, PRODUCT_NAME) is None


def test_find_extracted_safe_dir_supports_legacy_nested_cache(tmp_path):
    products_dir = tmp_path / "products"
    product_stem = PRODUCT_NAME.removesuffix(".SAFE")
    legacy_safe = _mark_complete_safe(products_dir / product_stem / PRODUCT_NAME)

    assert _find_extracted_safe_dir(products_dir, PRODUCT_NAME) == legacy_safe


def test_download_extracts_safe_without_duplicate_product_stem(tmp_path):
    cache_dir = tmp_path / "cache"
    products_dir = cache_dir / "products"
    products_dir.mkdir(parents=True)

    product_stem = PRODUCT_NAME.removesuffix(".SAFE")
    zip_path = products_dir / f"{product_stem}.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{PRODUCT_NAME}/MTD_MSIL1C.xml", "<xml />")
        zf.writestr(f"{PRODUCT_NAME}/GRANULE/example.txt", "ok")

    safe_dir = download_cdse_product_zip(
        product_id="unused-because-zip-already-exists",
        product_name=PRODUCT_NAME,
        access_token="unused",
        cache_dir=cache_dir,
        verbose=False,
    )

    expected = products_dir / PRODUCT_NAME
    duplicated = products_dir / product_stem / PRODUCT_NAME

    assert safe_dir == expected
    assert (safe_dir / "MTD_MSIL1C.xml").is_file()
    assert not duplicated.exists()
