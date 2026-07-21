from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional
import shutil
import zipfile

import requests
import re

from ..sys_cfg import DATA_PATH
from .constants import (
    PHISAT2_L0_COLLECTION,
    PHISAT2_L1_COLLECTION,
    PHISAT2_L1C_COLLECTION,
    PHISAT2_L1A_COLLECTION,
    DEFAULT_L0_DOWNLOAD_DIR,
    DEFAULT_L1_DOWNLOAD_DIR,
    DEFAULT_L1A_DOWNLOAD_DIR,
    VM_L0_ROOT,
    VM_L1_ROOT,
    VM_L1A_ROOT,
)
from phiesta.l1a import L1A_event

from .catalog_geometry import (
    catalog_geo_from_feature,
    bbox_to_wkt,
    point_buffer_to_wkt,
)

@dataclass
class InsulaPaths:
    zip_path: Path
    extract_dir: Path
    product_folder: Path

def _normalize_identifier_query(identifier: str) -> str:
    s = str(identifier).strip()
    if s.endswith(".zip"):
        s = s[:-4]
    return s


def _extract_acq_id(text: str | None) -> Optional[str]:
    if text is None:
        return None
    s = str(text)
    m = re.search(r"PHISAT-2_L[0-9A-Z]+_(\d+)_", s)
    if m is None:
        return None
    out = m.group(1).lstrip("0")
    return out if out else "0"

def _day_bounds_utc(date_str: str) -> tuple[str, str]:
    """
    Convert YYYY-MM-DD into a UTC day interval.

    Returns:
        ("YYYY-MM-DDT00:00:00Z", "YYYY-MM-DDT23:59:59Z")
    """
    d = str(date_str).strip()
    if len(d) != 10:
        raise ValueError("date must be in YYYY-MM-DD format")
    return (f"{d}T00:00:00Z", f"{d}T23:59:59Z")

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
    



    def export_search_table_geojson(self, table, out_geojson):
        """
        Export a compact search table to GeoJSON.

        This is useful to visualize search results in QGIS, geojson.io,
        notebooks, or map-based tools.
        """
        from phiesta.remote.search_table import export_search_table_geojson

        return export_search_table_geojson(table, out_geojson)


    def load_l1_table(
        self,
        table,
        limit=None,
        product_id_col="product_id",
        continue_on_error=True,
        **load_kwargs,
    ):
        """
        Load/download L1 products listed in a compact search table.
        """
        from phiesta.remote.search_table import load_products_from_table

        return load_products_from_table(
            self,
            table,
            level="L1",
            limit=limit,
            product_id_col=product_id_col,
            continue_on_error=continue_on_error,
            **load_kwargs,
        )


    def load_l0_table(
        self,
        table,
        limit=None,
        product_id_col="product_id",
        continue_on_error=True,
        **load_kwargs,
    ):
        """
        Load/download L0 products listed in a compact search table.
        """
        from phiesta.remote.search_table import load_products_from_table

        return load_products_from_table(
            self,
            table,
            level="L0",
            limit=limit,
            product_id_col=product_id_col,
            continue_on_error=continue_on_error,
            **load_kwargs,
        )


    def search_l1_bbox_table(
        self,
        bbox_lonlat,
        pages=40,
        results_per_page=100,
        **search_kwargs,
    ):
        """
        Search L1 products intersecting a lon/lat bbox and return a compact DataFrame.

        Args:
            bbox_lonlat: (min_lon, min_lat, max_lon, max_lat).
            pages: maximum number of Insula pages to scan.
            results_per_page: number of products per page.
            **search_kwargs: extra arguments forwarded to search_l1.
        """
        from phiesta.remote.search_table import search_bbox_table

        return search_bbox_table(
            self,
            level="L1",
            bbox_lonlat=bbox_lonlat,
            pages=pages,
            results_per_page=results_per_page,
            **search_kwargs,
        )


    def search_l0_bbox_table(
        self,
        bbox_lonlat,
        pages=40,
        results_per_page=100,
        **search_kwargs,
    ):
        """
        Search L0 products intersecting a lon/lat bbox and return a compact DataFrame.

        Args:
            bbox_lonlat: (min_lon, min_lat, max_lon, max_lat).
            pages: maximum number of Insula pages to scan.
            results_per_page: number of products per page.
            **search_kwargs: extra arguments forwarded to search_l0.
        """
        from phiesta.remote.search_table import search_bbox_table

        return search_bbox_table(
            self,
            level="L0",
            bbox_lonlat=bbox_lonlat,
            pages=pages,
            results_per_page=results_per_page,
            **search_kwargs,
        )


    def search_l1_table(self, *args, **kwargs):
        """
        Search L1 products and return a compact pandas DataFrame.

        The table includes product id, filename, acquisition datetime, center,
        and footprint corners when available.
        """
        from phiesta.remote.search_table import search_result_to_dataframe

        result = self.search_l1(*args, **kwargs)
        return search_result_to_dataframe(result)


    def search_l0_table(self, *args, **kwargs):
        """
        Search L0 products and return a compact pandas DataFrame.

        The table includes product id, filename, acquisition datetime, center,
        and footprint corners when available.
        """
        from phiesta.remote.search_table import search_result_to_dataframe

        result = self.search_l0(*args, **kwargs)
        return search_result_to_dataframe(result)


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
    
    def load_l1c(self, identifier, **kwargs):
        """
        Explicit alias for the current ΦSat-2 L1C-like products.

        `load_l1(...)` is kept as a backward-compatible alias.
        """
        return self.load_l1(identifier, **kwargs)

    def load_l1a(self, identifier, **kwargs):
        """
        Load a ΦSat-2 L1A product from Insula.

        This requires PHISAT2_L1A_COLLECTION to be configured with the
        corresponding Insula refDataCollection id.
        """
        if PHISAT2_L1A_COLLECTION is None:
            raise RuntimeError(
                "PHISAT2_L1A_COLLECTION is not configured yet. "
                "Set the Insula L1A refDataCollection id in "
                "phiesta.remote.constants before calling load_l1a(...)."
            )

        kwargs.setdefault("dest_dir", DEFAULT_L1A_DOWNLOAD_DIR)

        return L1A_event.from_insula_identifier(
            identifier=identifier,
            client=self,
            ref_data_collection=PHISAT2_L1A_COLLECTION,
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
    
    def download_l1c(self, identifier=None, **kwargs):
        """
        Explicit alias for downloading the current ΦSat-2 L1C-like products.

        `download_l1(...)` is kept as a backward-compatible alias.
        """
        return self.download_l1(identifier=identifier, **kwargs)

    def download_l1a(self, identifier=None, **kwargs):
        """
        Download a ΦSat-2 L1A product from Insula.

        This requires PHISAT2_L1A_COLLECTION to be configured with the
        corresponding Insula refDataCollection id.
        """
        if PHISAT2_L1A_COLLECTION is None:
            raise RuntimeError(
                "PHISAT2_L1A_COLLECTION is not configured yet. "
                "Set the Insula L1A refDataCollection id in "
                "phiesta.remote.constants before calling download_l1a(...)."
            )

        return self.download_ref_data(
            identifier=identifier,
            ref_data_collection=PHISAT2_L1A_COLLECTION,
            **kwargs,
        )

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
        """
        Retrieve exactly one REF_DATA feature matching the requested identifier.

        Accepted user inputs:
        - bare acquisition id, e.g. "4520"
        - product name without .zip
        - full productIdentifier / filename
        """
        q = _normalize_identifier_query(identifier)
        q_acq = _extract_acq_id(q)
        if q_acq is None and q.isdigit():
            q_acq = q.lstrip("0") or "0"

        data = self.search_ref_data(
            ref_data_collection=ref_data_collection,
            page=0,
            results_per_page=100,
            identifier=q,
        )
        features = data.get("features", [])
        if not features:
            raise ValueError(f"No feature found for identifier={identifier!r}")

        exact = []

        for feat in features:
            props = feat.get("properties", {})

            pid = props.get("productIdentifier")
            fn = props.get("filename")

            pid_n = _normalize_identifier_query(pid) if pid else None
            fn_n = _normalize_identifier_query(fn) if fn else None

            pid_acq = _extract_acq_id(pid)
            fn_acq = _extract_acq_id(fn)

            candidates = {x for x in [pid_n, fn_n, pid_acq, fn_acq] if x}

            if q in candidates or (q_acq is not None and q_acq in candidates):
                exact.append(feat)

        if len(exact) == 1:
            return exact[0]

        if len(exact) == 0:
            sample = [
                f.get("properties", {}).get("productIdentifier")
                for f in features[:10]
            ]
            raise ValueError(
                f"No exact feature match found for identifier={identifier!r}. "
                f"Top search results were: {sample}"
            )

        sample = [
            f.get("properties", {}).get("productIdentifier")
            for f in exact[:10]
        ]
        raise ValueError(
            f"Multiple exact feature matches found for identifier={identifier!r}: {sample}"
        )

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
    
    def get_catalog_geo(
        self,
        ref_data_collection: str,
        identifier: str,
    ) -> Dict[str, Any]:
        """
        Return the lightweight catalog geometry object for one Insula feature.
        """
        feature = self.get_feature_by_identifier(
            ref_data_collection=ref_data_collection,
            identifier=identifier,
        )
        return catalog_geo_from_feature(
            feature=feature,
            ref_data_collection=ref_data_collection,
        )


    def get_l0_catalog_geo(self, identifier: str) -> Dict[str, Any]:
        """
        Convenience wrapper for PHISAT-2 L0 catalog geometry.
        """
        return self.get_catalog_geo(
            ref_data_collection=PHISAT2_L0_COLLECTION,
            identifier=identifier,
        )


    def get_l1_catalog_geo(self, identifier: str) -> Dict[str, Any]:
        """
        Convenience wrapper for PHISAT-2 L1 catalog geometry.
        """
        return self.get_catalog_geo(
            ref_data_collection=PHISAT2_L1_COLLECTION,
            identifier=identifier,
        )
    
    def search_l1_in_aoi(
        self,
        aoi_wkt: str,
        date: str | None = None,
        product_date_start: str | None = None,
        product_date_end: str | None = None,
        page: int = 0,
        results_per_page: int = 20,
        **extra_params: Any,
    ) -> list[Dict[str, Any]]:
        """
        Search PHISAT-2 L1 products intersecting a given AOI.

        Args:
            aoi_wkt: AOI expressed as a WKT polygon string in lon/lat order.
            date: Optional shortcut for a whole UTC day in YYYY-MM-DD format.
            product_date_start: Optional explicit UTC start datetime.
            product_date_end: Optional explicit UTC end datetime.
            page: Page index.
            results_per_page: Requested page size.
            **extra_params: Additional Insula search parameters.

        Returns:
            A list of Insula feature dictionaries.

        Raises:
            ValueError: If both `date` and explicit date bounds are provided.
        """
        if date is not None and (product_date_start is not None or product_date_end is not None):
            raise ValueError(
                "Use either `date=...` or (`product_date_start`, `product_date_end`), not both."
            )

        if date is not None:
            product_date_start, product_date_end = _day_bounds_utc(date)

        params: Dict[str, Any] = {
            "aoi": aoi_wkt,
        }
        if product_date_start is not None:
            params["productDateStart"] = product_date_start
        if product_date_end is not None:
            params["productDateEnd"] = product_date_end
        params.update(extra_params)

        data = self.search_ref_data(
            ref_data_collection=PHISAT2_L1_COLLECTION,
            page=page,
            results_per_page=results_per_page,
            **params,
        )
        return data.get("features", [])


    def search_l0_in_aoi(
        self,
        aoi_wkt: str,
        date: str | None = None,
        product_date_start: str | None = None,
        product_date_end: str | None = None,
        page: int = 0,
        results_per_page: int = 20,
        **extra_params: Any,
    ) -> list[Dict[str, Any]]:
        """
        Search PHISAT-2 L0 products intersecting a given AOI.
        """
        if date is not None and (product_date_start is not None or product_date_end is not None):
            raise ValueError(
                "Use either `date=...` or (`product_date_start`, `product_date_end`), not both."
            )

        if date is not None:
            product_date_start, product_date_end = _day_bounds_utc(date)

        params: Dict[str, Any] = {
            "aoi": aoi_wkt,
        }
        if product_date_start is not None:
            params["productDateStart"] = product_date_start
        if product_date_end is not None:
            params["productDateEnd"] = product_date_end
        params.update(extra_params)

        data = self.search_ref_data(
            ref_data_collection=PHISAT2_L0_COLLECTION,
            page=page,
            results_per_page=results_per_page,
            **params,
        )
        return data.get("features", [])


    def search_l1_in_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        date: str | None = None,
        product_date_start: str | None = None,
        product_date_end: str | None = None,
        page: int = 0,
        results_per_page: int = 20,
        **extra_params: Any,
    ) -> list[Dict[str, Any]]:
        """
        Search PHISAT-2 L1 products intersecting a bounding box.
        """
        aoi_wkt = bbox_to_wkt(min_lon, min_lat, max_lon, max_lat)
        return self.search_l1_in_aoi(
            aoi_wkt=aoi_wkt,
            date=date,
            product_date_start=product_date_start,
            product_date_end=product_date_end,
            page=page,
            results_per_page=results_per_page,
            **extra_params,
        )


    def search_l0_in_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        date: str | None = None,
        product_date_start: str | None = None,
        product_date_end: str | None = None,
        page: int = 0,
        results_per_page: int = 20,
        **extra_params: Any,
    ) -> list[Dict[str, Any]]:
        """
        Search PHISAT-2 L0 products intersecting a bounding box.
        """
        aoi_wkt = bbox_to_wkt(min_lon, min_lat, max_lon, max_lat)
        return self.search_l0_in_aoi(
            aoi_wkt=aoi_wkt,
            date=date,
            product_date_start=product_date_start,
            product_date_end=product_date_end,
            page=page,
            results_per_page=results_per_page,
            **extra_params,
        )


    def search_l1_near_point(
        self,
        lon: float,
        lat: float,
        radius_km: float,
        date: str | None = None,
        product_date_start: str | None = None,
        product_date_end: str | None = None,
        page: int = 0,
        results_per_page: int = 20,
        n_points: int = 64,
        **extra_params: Any,
    ) -> list[Dict[str, Any]]:
        """
        Search PHISAT-2 L1 products around a lon/lat point using a circular AOI approximation.
        """
        aoi_wkt = point_buffer_to_wkt(
            lon=lon,
            lat=lat,
            radius_km=radius_km,
            n_points=n_points,
        )
        return self.search_l1_in_aoi(
            aoi_wkt=aoi_wkt,
            date=date,
            product_date_start=product_date_start,
            product_date_end=product_date_end,
            page=page,
            results_per_page=results_per_page,
            **extra_params,
        )


    def search_l0_near_point(
        self,
        lon: float,
        lat: float,
        radius_km: float,
        date: str | None = None,
        product_date_start: str | None = None,
        product_date_end: str | None = None,
        page: int = 0,
        results_per_page: int = 20,
        n_points: int = 64,
        **extra_params: Any,
    ) -> list[Dict[str, Any]]:
        """
        Search PHISAT-2 L0 products around a lon/lat point using a circular AOI approximation.
        """
        aoi_wkt = point_buffer_to_wkt(
            lon=lon,
            lat=lat,
            radius_km=radius_km,
            n_points=n_points,
        )
        return self.search_l0_in_aoi(
            aoi_wkt=aoi_wkt,
            date=date,
            product_date_start=product_date_start,
            product_date_end=product_date_end,
            page=page,
            results_per_page=results_per_page,
            **extra_params,
        )
