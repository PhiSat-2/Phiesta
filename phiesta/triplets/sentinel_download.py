from __future__ import annotations

import getpass
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import requests


CATALOGUE_PRODUCTS_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_PRODUCTS_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def _safe_name_from_path(path: str) -> str:
    return Path(str(path).rstrip("/")).name


def _escape_odata_string(value: str) -> str:
    return value.replace("'", "''")


def get_cdse_access_token(
    username: str | None = None,
    password: str | None = None,
    *,
    verbose: bool = True,
) -> str:
    """
    Get a Copernicus Data Space Ecosystem access token.

    Order:
    - explicit username/password;
    - env vars CDSE_USERNAME / CDSE_PASSWORD;
    - interactive prompt.
    """
    username = username or os.environ.get("CDSE_USERNAME")
    password = password or os.environ.get("CDSE_PASSWORD")

    if username is None:
        username = input("CDSE username/email: ")

    if password is None:
        password = getpass.getpass("CDSE password: ")

    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Failed to obtain CDSE access token: "
            f"{response.status_code} {response.text[:500]}"
        )

    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("CDSE token response does not contain access_token.")

    if verbose:
        print("[CDSE] Access token acquired.")

    return token


def find_cdse_product_by_name(
    product_name: str,
    *,
    session: requests.Session | None = None,
) -> dict:
    """
    Find a CDSE product by exact Name and return its catalogue record.
    """
    session = session or requests.Session()

    name = _escape_odata_string(product_name)
    query = f"Name eq '{name}'"

    res = session.get(
        CATALOGUE_PRODUCTS_URL,
        params={
            "$filter": query,
            "$select": "Id,Name,S3Path,ContentDate",
            "$top": 5,
        },
        timeout=60,
    )
    res.raise_for_status()

    values = res.json().get("value", [])

    if not values:
        raise FileNotFoundError(f"Could not find CDSE product by Name: {product_name}")

    return values[0]


def _find_extracted_safe_dir(extract_dir: Path, product_name: str) -> Path | None:
    def is_complete_l1c_safe(path: Path) -> bool:
        # A failed ZIP extraction can leave the SAFE directory behind. Do not
        # treat that partial directory as reusable on the next run.
        return path.is_dir() and (path / "MTD_MSIL1C.xml").is_file()

    direct = extract_dir / product_name
    if is_complete_l1c_safe(direct):
        return direct

    # Backward compatibility with caches created by older Phiesta versions,
    # which extracted each archive inside an additional product-stem directory.
    # Search for the exact requested SAFE rather than returning an arbitrary one
    # when several Sentinel products share the same cache.
    for candidate in extract_dir.rglob(product_name):
        if is_complete_l1c_safe(candidate):
            return candidate

    return None


def download_cdse_product_zip(
    product_id: str,
    product_name: str,
    *,
    access_token: str,
    cache_dir: str | Path = "cache/sentinel2",
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Download a complete Sentinel product using CDSE OData.

    Returns path to extracted .SAFE directory.
    """
    cache_dir = Path(cache_dir)
    products_dir = cache_dir / "products"
    products_dir.mkdir(parents=True, exist_ok=True)

    product_stem = product_name[:-5] if product_name.endswith(".SAFE") else product_name
    zip_path = products_dir / f"{product_stem}.zip"

    # Sentinel ZIPs already contain a top-level ``<product>.SAFE`` directory.
    # Extracting into ``products/<product-stem>/`` duplicated the very long
    # product name in every path and can exceed Windows MAX_PATH. Extract
    # directly into ``products/`` instead; different products remain isolated
    # by their unique SAFE directory names.
    extract_dir = products_dir

    existing_safe = _find_extracted_safe_dir(extract_dir, product_name)
    if existing_safe is not None and not overwrite:
        if verbose:
            print(f"[CDSE] Reusing extracted SAFE: {existing_safe}")
        return existing_safe

    if not zip_path.exists() or overwrite:
        url = f"{DOWNLOAD_PRODUCTS_URL}({product_id})/$value"
        headers = {"Authorization": f"Bearer {access_token}"}

        if verbose:
            print(f"[CDSE] Downloading {product_name}")
            print(f"[CDSE] product_id={product_id}")
            print(f"[CDSE] output={zip_path}")

        with requests.Session() as session:
            session.headers.update(headers)

            with session.get(url, stream=True, allow_redirects=True, timeout=120) as response:
                if response.status_code != 200:
                    raise RuntimeError(
                        "CDSE download failed: "
                        f"{response.status_code} {response.text[:500]}"
                    )

                tmp_path = zip_path.with_suffix(".zip.tmp")
                with tmp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

                tmp_path.replace(zip_path)

    extract_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[CDSE] Extracting {zip_path} -> {extract_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    safe_dir = _find_extracted_safe_dir(extract_dir, product_name)
    if safe_dir is None:
        raise RuntimeError(f"Could not find extracted .SAFE directory in {extract_dir}")

    if verbose:
        print(f"[CDSE] Extracted SAFE: {safe_dir}")

    return safe_dir


def resolve_sentinel_l1c_safe_paths(
    source: Any,
    *,
    backend: str = "auto",
    cache_dir: str | Path = "cache/sentinel2",
    cdse_username: str | None = None,
    cdse_password: str | None = None,
    cdse_access_token: str | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[str]:
    """
    Resolve source.l1c_paths into local .SAFE directories.

    backend:
    - local_safe: require source.l1c_paths to exist locally.
    - download: download missing products from CDSE.
    - auto: use local SAFE if available, otherwise download.
    """
    backend = str(backend).lower()
    if backend not in {"auto", "local_safe", "download"}:
        raise ValueError("backend must be one of: auto, local_safe, download")

    l1c_paths = list(getattr(source, "l1c_paths", []) or [])
    if not l1c_paths:
        raise ValueError("SentinelSource has no l1c_paths.")

    local_paths = []
    missing = []

    for p in l1c_paths:
        path = Path(str(p))
        if path.exists():
            local_paths.append(str(path))
        else:
            missing.append(str(p))

    if not missing:
        if verbose:
            print("[CDSE] All Sentinel L1C SAFE paths are local.")
        return local_paths

    if backend == "local_safe":
        raise FileNotFoundError(
            "Some Sentinel L1C SAFE paths are not accessible locally:\n"
            + "\n".join(missing)
        )

    token = cdse_access_token
    if token is None:
        token = get_cdse_access_token(
            username=cdse_username,
            password=cdse_password,
            verbose=verbose,
        )

    session = requests.Session()

    # Metadata pairs produced by find_best_sentinel_source, after the next patch.
    pairs = []
    metadata = getattr(source, "metadata", None) or {}
    if isinstance(metadata, dict):
        pairs = metadata.get("pairs", []) or []

    resolved = list(local_paths)

    for missing_path in missing:
        product_name = _safe_name_from_path(missing_path)

        pair = None
        for p in pairs:
            if p.get("l1c_name") == product_name or _safe_name_from_path(p.get("l1c", "")) == product_name:
                pair = p
                break

        if pair is not None and pair.get("l1c_id"):
            product_id = pair["l1c_id"]
        else:
            record = find_cdse_product_by_name(product_name, session=session)
            product_id = record["Id"]
            product_name = record["Name"]

        safe_dir = download_cdse_product_zip(
            product_id=product_id,
            product_name=product_name,
            access_token=token,
            cache_dir=cache_dir,
            overwrite=overwrite,
            verbose=verbose,
        )

        resolved.append(str(safe_dir))

    return resolved
