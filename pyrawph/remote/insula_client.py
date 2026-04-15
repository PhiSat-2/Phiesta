from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional
import shutil
import zipfile

import requests


@dataclass
class InsulaPaths:
    zip_path: Path
    extract_dir: Path
    product_folder: Path


class InsulaClient:
    """
    Thin helper around Insula search/download for PHISAT-2 reference data.
    Handles:
    - auth header retrieval
    - REF_DATA search
    - pagination
    - zip download
    - extraction to cache
    """

    def __init__(
        self,
        insula_auth: Any,
        base_url: str = "https://phisat2.insula.earth",
        cache_dir: str | Path = "./insula_cache",
        timeout_search: int = 60,
        timeout_download: int = 300,
        verify_ssl: bool = True,
    ) -> None:
        self.insula_auth = insula_auth
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_search = int(timeout_search)
        self.timeout_download = int(timeout_download)
        self.verify_ssl = bool(verify_ssl)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": self.insula_auth.get_authorization_header()}

    def search_ref_data(
        self,
        ref_data_collection: str,
        page: int = 0,
        results_per_page: int = 20,
        **extra_params: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "catalogue": "REF_DATA",
            "refDataCollection": ref_data_collection,
            "page": int(page),
            "resultsPerPage": int(results_per_page),
        }
        params.update(extra_params)

        resp = requests.get(
            f"{self.base_url}/secure/api/v2.0/search",
            headers=self._headers(),
            params=params,
            verify=self.verify_ssl,
            timeout=self.timeout_search,
        )
        resp.raise_for_status()
        return resp.json()

    def iter_ref_data(
        self,
        ref_data_collection: str,
        results_per_page: int = 50,
        max_pages: Optional[int] = None,
        **extra_params: Any,
    ) -> Generator[Dict[str, Any], None, None]:
        page = 0
        seen_pages = 0

        while True:
            data = self.search_ref_data(
                ref_data_collection=ref_data_collection,
                page=page,
                results_per_page=results_per_page,
                **extra_params,
            )

            features = data.get("features", [])
            if not features:
                break

            for feature in features:
                yield feature

            seen_pages += 1
            total_pages = data.get("page", {}).get("totalPages", None)

            if max_pages is not None and seen_pages >= max_pages:
                break
            if total_pages is not None and page + 1 >= total_pages:
                break

            page += 1

    def get_feature_by_identifier(
        self,
        ref_data_collection: str,
        identifier: str,
    ) -> Dict[str, Any]:
        data = self.search_ref_data(
            ref_data_collection=ref_data_collection,
            page=0,
            results_per_page=20,
            identifier=identifier,
        )
        features = data.get("features", [])
        if not features:
            raise ValueError(f"No feature found for identifier={identifier!r}")
        return features[0]

    def _paths_from_feature(self, feature: Dict[str, Any]) -> InsulaPaths:
        props = feature["properties"]
        filename = props["filename"]

        zip_path = self.cache_dir / filename
        extract_dir = self.cache_dir / Path(filename).stem

        product_folder = extract_dir
        return InsulaPaths(
            zip_path=zip_path,
            extract_dir=extract_dir,
            product_folder=product_folder,
        )

    def download_feature(
        self,
        feature: Dict[str, Any],
        *,
        extract: bool = True,
        keep_zip: bool = False,
        skip_existing: bool = True,
        force_redownload: bool = False,
    ) -> Path:
        props = feature["properties"]
        download_url = props["_links"]["download"]["href"]

        paths = self._paths_from_feature(feature)

        if skip_existing and paths.extract_dir.exists() and any(paths.extract_dir.iterdir()):
            children = list(paths.extract_dir.iterdir())
            if len(children) == 1 and children[0].is_dir():
                return children[0]
            return paths.extract_dir

        if force_redownload:
            if paths.zip_path.exists():
                paths.zip_path.unlink()
            if paths.extract_dir.exists():
                shutil.rmtree(paths.extract_dir)

        with requests.get(
            download_url,
            headers=self._headers(),
            stream=True,
            verify=self.verify_ssl,
            timeout=self.timeout_download,
        ) as resp:
            resp.raise_for_status()
            with open(paths.zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        if not extract:
            return paths.zip_path

        paths.extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(paths.zip_path, "r") as zf:
            zf.extractall(paths.extract_dir)

        if not keep_zip and paths.zip_path.exists():
            paths.zip_path.unlink()

        children = list(paths.extract_dir.iterdir())
        if len(children) == 1 and children[0].is_dir():
            return children[0]
        return paths.extract_dir