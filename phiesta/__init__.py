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
    product_gallery,
    compare_processing_switches,
    compare_levels,
)

from phiesta.geometry import (
    interband_shift_table,
)

from phiesta.specs import (
    phisat2_band_table,
    phisat2_product_level_specs,
    mission_spec_report,
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
    "product_gallery",
    "compare_processing_switches",
    "compare_levels",
    "interband_shift_table",
    "phisat2_band_table",
    "phisat2_product_level_specs",
    "mission_spec_report",
]
