from .insula_client import InsulaClient
from .constants import (
    PHISAT2_BASE_URL,
    PHISAT2_L0_COLLECTION,
    PHISAT2_L1_COLLECTION,
    PHISAT2_L1C_COLLECTION,
    PHISAT2_L1A_COLLECTION,
    DEFAULT_DOWNLOAD_DIR,
    DEFAULT_L0_DOWNLOAD_DIR,
    DEFAULT_L1_DOWNLOAD_DIR,
    VM_L0_ROOT,
    VM_L1_ROOT,
)
from .auth import connect_insula
from .catalog_geometry import (
    catalog_geo_from_feature,
    enrich_meta_with_insula_feature,
    get_catalog_corners,
    get_catalog_center,
    get_catalog_polygon,
    format_catalog_geo,
    print_catalog_geo,
    get_catalog_identifier,
    get_catalog_acquisition_datetime,
    bbox_to_wkt,
    point_buffer_to_wkt,
    get_catalog_bbox_lonlat,
    extract_phisat_acquisition_id
)

__all__ = [
    "InsulaClient",
    "connect_insula",
    "PHISAT2_BASE_URL",
    "PHISAT2_L0_COLLECTION",
    "PHISAT2_L1_COLLECTION",
    "PHISAT2_L1C_COLLECTION",
    "PHISAT2_L1A_COLLECTION",
    "DEFAULT_DOWNLOAD_DIR",
    "DEFAULT_L0_DOWNLOAD_DIR",
    "DEFAULT_L1_DOWNLOAD_DIR",
    "VM_L0_ROOT",
    "VM_L1_ROOT",
    "catalog_geo_from_feature",
    "enrich_meta_with_insula_feature",
    "get_catalog_corners",
    "get_catalog_center",
    "get_catalog_polygon",
    "format_catalog_geo",
    "print_catalog_geo",
    "get_catalog_identifier",
    "get_catalog_acquisition_datetime",
    "bbox_to_wkt",
    "point_buffer_to_wkt",
    "get_catalog_bbox_lonlat",
    "extract_phisat_acquisition_id",
]
from .search_table import search_result_to_dataframe, search_result_to_records, export_search_result_csv
from .search_table import add_footprint_bbox_columns, filter_dataframe_by_bbox, search_bbox_table
from .search_table import search_table_to_geojson, export_search_table_geojson, load_products_from_table


from .worldcover import (
    WORLDCOVER_CLASSES,
    resolve_worldcover_class,
    worldcover_stats_for_feature,
    search_l1_worldcover,
)
