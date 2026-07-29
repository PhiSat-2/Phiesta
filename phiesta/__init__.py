from phiesta.remote.auth import connect_insula

from phiesta.l0 import raw_l0_report, raw_l0_table

from phiesta.products import (
    open_product,
    open_raw_l0_product,
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
    "raw_l0_report",
    "raw_l0_table",
    "open_product",
    "open_raw_l0_product",
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
