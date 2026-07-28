from phiesta.remote.auth import connect_insula

from phiesta.products import (
    open_product,
    classify_product_file,
    file_manifest,
    file_family_summary,
    processing_switches,
    raster_inventory,
    product_card,
    compare_product_folders,
    quality_report,
    quality_table,
)

__all__ = [
    "connect_insula",
    "open_product",
    "classify_product_file",
    "file_manifest",
    "file_family_summary",
    "processing_switches",
    "raster_inventory",
    "product_card",
    "compare_product_folders",
    "quality_report",
    "quality_table",
]
