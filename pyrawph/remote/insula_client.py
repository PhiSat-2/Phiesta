from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional
import shutil
import zipfile

import requests

from ..sys_cfg import DATA_PATH
from .constants import (
    PHISAT2_L0_COLLECTION,
    PHISAT2_L1_COLLECTION,
    DEFAULT_L0_DOWNLOAD_DIR,
    DEFAULT_L1_DOWNLOAD_DIR,
    VM_L0_ROOT,
    VM_L1_ROOT,
)

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
        cache_dir: str | Path = DATA_PATH,
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
    

    def _resolve_or_download_product_folder(
        self,
        *,
        ref_data_collection: str,
        identifier: str,
        default_dest_dir: str | Path,
        vm_root: str | Path | None = None,
        dest_dir: str | Path | None = None,
        local_fallback: bool = True,
        vm_fallback: bool = False,
        local_roots: Optional[list[str | Path]] = None,
        keep_zip: bool = False,
        skip_existing: bool = True,
        force_redownload: bool = False,
    ) -> tuple[Path, Optional[Dict[str, Any]]]:
        """
        Resolve a product folder from local / VM paths, or download it from Insula.

        Returns:
            `(product_folder, feature_props)` where `feature_props` is `None` if the
            folder was resolved locally/VM, and otherwise the downloaded feature
            properties dictionary.
        """
        from .local_resolver import resolve_existing_product

        search_roots: list[str | Path] = []
        if local_fallback:
            search_roots.append(dest_dir or default_dest_dir)
        if vm_fallback and vm_root is not None:
            search_roots.append(vm_root)
        if local_roots:
            search_roots.extend(local_roots)

        if not force_redownload:
            existing = resolve_existing_product(
                identifier=identifier,
                roots=search_roots,
            )
            if existing is not None:
                return Path(existing), None

        feature = self.get_feature_by_identifier(
            ref_data_collection=ref_data_collection,
            identifier=identifier,
        )
        folder = Path(
            self.download_feature(
                feature,
                dest_dir=dest_dir or default_dest_dir,
                extract=True,
                keep_zip=keep_zip,
                skip_existing=skip_existing,
                force_redownload=force_redownload,
            )
        )
        return folder, feature["properties"]
    
    def search_l1(self, date: str | None = None, page: int = 0, results_per_page: int = 5) -> list[Dict[str, Any]]:
        """
        Search PHISAT-2 L1 reference-data products.

        If `date` is provided, all L1 products matching that YYYY-MM-DD date are
        returned. Otherwise, one page of generic Insula search results is returned.

        Args:
            date: Optional acquisition date in YYYY-MM-DD format.
            page: Page index used for generic search when `date` is not provided.
            results_per_page: Requested page size for generic search.

        Returns:
            A list of Insula feature dictionaries.
        """

        if date is not None:
            return self.search_ref_data_by_date(
                ref_data_collection=PHISAT2_L1_COLLECTION,
                date=date,
                results_per_page=100,
            )

        data = self.search_ref_data(
            ref_data_collection=PHISAT2_L1_COLLECTION,
            page=page,
            results_per_page=results_per_page,
        )
        return data["features"]
    
    def search_l0(self, date: str | None = None, page: int = 0, results_per_page: int = 5) -> list[Dict[str, Any]]:
        """
        Search PHISAT-2 L0 reference-data products.

        If `date` is provided, all L0 products matching that YYYY-MM-DD date are
        returned. Otherwise, one page of generic Insula search results is returned.

        Args:
            date: Optional acquisition date in YYYY-MM-DD format.
            page: Page index used for generic search when `date` is not provided.
            results_per_page: Requested page size for generic search.

        Returns:
            A list of Insula feature dictionaries.
        """

        if date is not None:
            return self.search_ref_data_by_date(
                ref_data_collection=PHISAT2_L0_COLLECTION,
                date=date,
                results_per_page=100,
            )

        data = self.search_ref_data(
            ref_data_collection=PHISAT2_L0_COLLECTION,
            page=page,
            results_per_page=results_per_page,
        )
        return data["features"]
    
    def load_l1(
        self,
        identifier: str,
        vm_fallback: bool = True,
        **kwargs,
    ) -> "L1_event":
        """
        Load one PHISAT-2 L1 product by acquisition identifier.

        The loader first checks local data folders and, optionally, the shared VM
        directory before downloading the product from Insula.

        Args:
            identifier: Acquisition id or full product identifier.
            vm_fallback: If True, also search the shared VM L1 directory.
            **kwargs: Additional options forwarded to `L1_event.from_insula_identifier()`.

        Returns:
            A loaded `L1_event`.
        """
        from ..l1.l1_event import L1_event

        return L1_event.from_insula_identifier(
            client=self,
            ref_data_collection=PHISAT2_L1_COLLECTION,
            identifier=identifier,
            vm_fallback=vm_fallback,
            **kwargs,
        )
    
    def load_l0(
        self,
        identifier: str,
        vm_fallback: bool = True,
        convert: bool = True,
        **kwargs,
    ) -> "L0_event | Path":
        """
        Load one PHISAT-2 L0 product by acquisition identifier.

        The loader first checks local data folders and, optionally, the shared VM
        directory before downloading the product from Insula.

        Args:
            identifier: Acquisition id or full product identifier.
            vm_fallback: If True, also search the shared VM L0 directory.
            convert: If True, convert `raw.bin` into TIFF bands and return an
                `L0_event`. If False, return the resolved product folder path.
            **kwargs: Additional options forwarded to `L0_event.from_insula_identifier()`.

        Returns:
            An `L0_event` if `convert=True`, otherwise a `Path`.
        """
        from ..l0.l0_event import L0_event

        return L0_event.from_insula_identifier(
            client=self,
            ref_data_collection=PHISAT2_L0_COLLECTION,
            identifier=identifier,
            vm_fallback=vm_fallback,
            convert=convert,
            **kwargs,
        )
    
    def download_l1(
        self,
        identifier: str | None = None,
        date: str | None = None,
        all_matches: bool = False,
        vm_fallback: bool = True,
        **kwargs,
    ) -> Path | list[Path]:
        """
        Download PHISAT-2 L1 products without loading them as `L1_event`.

        Args:
            identifier: One acquisition id / product identifier to download.
            date: Date in YYYY-MM-DD format. Use with `all_matches=True`.
            all_matches: If True and `date` is provided, download all matching L1
                products for that day.
            vm_fallback: If True, reuse products already present in the shared VM L1
                directory.
            **kwargs: Additional options forwarded to the internal folder-resolution
                helper.

        Returns:
            One product folder path, or a list of folder paths.
        """
        if identifier is None and date is None:
            raise ValueError("Provide either `identifier` or `date`.")
        if date is not None and not all_matches:
            raise ValueError(
                "For date-based downloads, use `search_l1(date=...)` then "
                "`download_l1(identifier=...)`, or set `all_matches=True`."
            )

        if date is not None:
            features = self.search_l1(date=date, results_per_page=100)
            folders: list[Path] = []
            for feature in features:
                ident = feature["properties"]["productIdentifier"]
                folder, _ = self._resolve_or_download_product_folder(
                    ref_data_collection=PHISAT2_L1_COLLECTION,
                    identifier=ident,
                    default_dest_dir=DEFAULT_L1_DOWNLOAD_DIR,
                    vm_root=VM_L1_ROOT,
                    vm_fallback=vm_fallback,
                    **kwargs,
                )
                folders.append(folder)
            return folders

        folder, _ = self._resolve_or_download_product_folder(
            ref_data_collection=PHISAT2_L1_COLLECTION,
            identifier=identifier,
            default_dest_dir=DEFAULT_L1_DOWNLOAD_DIR,
            vm_root=VM_L1_ROOT,
            vm_fallback=vm_fallback,
            **kwargs,
        )
        return folder
    
    def download_l0(
        self,
        identifier: str | None = None,
        date: str | None = None,
        all_matches: bool = False,
        vm_fallback: bool = True,
        convert: bool = False,
        **kwargs,
    ) -> Path | list[Path]:
        """
        Download PHISAT-2 L0 products.

        Args:
            identifier: One acquisition id / product identifier to download.
            date: Date in YYYY-MM-DD format. Use with `all_matches=True`.
            all_matches: If True and `date` is provided, download all matching L0
                products for that day.
            vm_fallback: If True, reuse products already present in the shared VM L0
                directory.
            convert: If True, convert the downloaded product into a prepared local L0
                layout before returning.
            **kwargs: Additional options forwarded to the loader/download helper.

        Returns:
            One product folder path, or a list of folder paths.
        """
        if identifier is None and date is None:
            raise ValueError("Provide either `identifier` or `date`.")
        if date is not None and not all_matches:
            raise ValueError(
                "For date-based downloads, use `search_l0(date=...)` then "
                "`download_l0(identifier=...)`, or set `all_matches=True`."
            )

        if convert:
            if date is not None:
                features = self.search_l0(date=date, results_per_page=100)
                folders: list[Path] = []
                for feature in features:
                    ident = feature["properties"]["productIdentifier"]
                    evt_or_folder = self.load_l0(
                        identifier=ident,
                        vm_fallback=vm_fallback,
                        convert=True,
                        **kwargs,
                    )
                    folders.append(Path(evt_or_folder.get_meta()["prepared_product_folder"]))
                return folders

            evt_or_folder = self.load_l0(
                identifier=identifier,
                vm_fallback=vm_fallback,
                convert=True,
                **kwargs,
            )
            return Path(evt_or_folder.get_meta()["prepared_product_folder"])

        if date is not None:
            features = self.search_l0(date=date, results_per_page=100)
            folders: list[Path] = []
            for feature in features:
                ident = feature["properties"]["productIdentifier"]
                folder, _ = self._resolve_or_download_product_folder(
                    ref_data_collection=PHISAT2_L0_COLLECTION,
                    identifier=ident,
                    default_dest_dir=DEFAULT_L0_DOWNLOAD_DIR,
                    vm_root=VM_L0_ROOT,
                    vm_fallback=vm_fallback,
                    **kwargs,
                )
                folders.append(folder)
            return folders

        folder, _ = self._resolve_or_download_product_folder(
            ref_data_collection=PHISAT2_L0_COLLECTION,
            identifier=identifier,
            default_dest_dir=DEFAULT_L0_DOWNLOAD_DIR,
            vm_root=VM_L0_ROOT,
            vm_fallback=vm_fallback,
            **kwargs,
        )
        return folder
    

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
    
    def search_ref_data_by_date(
        self,
        ref_data_collection: str,
        date: str,
        results_per_page: int = 100,
        max_pages: int | None = None,
        **extra_params,
    ) -> list[Dict[str, Any]]:
        """
        Return all remote features whose `startDate` matches a given YYYY-MM-DD date.

        The method assumes results are ordered from newest to oldest and stops early
        once all subsequent pages are guaranteed to be older than the target date.

        Args:
            ref_data_collection: Insula REF_DATA collection id.
            date: Target date in YYYY-MM-DD format.
            results_per_page: Requested page size.
            max_pages: Optional upper bound on the number of pages to inspect.
            **extra_params: Additional Insula search parameters.

        Returns:
            A list of matching Insula feature dictionaries.
        """
        target = str(date).strip()
        if not target:
            raise ValueError("date must be a non-empty string in YYYY-MM-DD format")

        out = []
        page = 0

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

            page_dates = [
                f.get("properties", {}).get("startDate", "")[:10]
                for f in features
            ]

            matched = [
                f for f in features
                if f.get("properties", {}).get("startDate", "")[:10] == target
            ]
            out.extend(matched)

            # We assume newest -> oldest ordering inside and across pages
            newest = page_dates[0]
            oldest = page_dates[-1]

            # If the newest item on this page is already older than target,
            # then all following pages will also be older.
            if newest < target:
                break

            # If this page crosses below the target date and contains no match,
            # then we have already passed the target day.
            if oldest < target and not matched:
                break

            total_pages = data.get("page", {}).get("totalPages", None)

            page += 1
            if max_pages is not None and page >= max_pages:
                break
            if total_pages is not None and page >= total_pages:
                break

        return out

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

    def _paths_from_feature(
        self,
        feature: Dict[str, Any],
        base_dir: str | Path | None = None,
    ) -> InsulaPaths:
        props = feature["properties"]
        filename = props["filename"]

        root_dir = Path(base_dir) if base_dir is not None else self.cache_dir
        root_dir.mkdir(parents=True, exist_ok=True)

        zip_path = root_dir / filename
        extract_dir = root_dir / Path(filename).stem
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
        dest_dir: str | Path | None = None,
        extract: bool = True,
        keep_zip: bool = False,
        skip_existing: bool = True,
        force_redownload: bool = False,
    ) -> Path:
        """
        Download one Insula feature and optionally extract it locally.

        Args:
            feature: One Insula feature dictionary returned by search.
            dest_dir: Optional destination directory. If `None`, use the client cache dir.
            extract: If True, extract the downloaded zip archive.
            keep_zip: If True, keep the zip archive after extraction.
            skip_existing: If True, reuse an existing extracted folder when possible.
            force_redownload: If True, delete any previous local copy and download again.

        Returns:
            The extracted product folder path if `extract=True`, otherwise the zip path.
        """
        props = feature["properties"]
        download_url = props["_links"]["download"]["href"]

        paths = self._paths_from_feature(feature, base_dir=dest_dir)

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