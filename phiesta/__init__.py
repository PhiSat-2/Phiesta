from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("phiesta")
except PackageNotFoundError:
    __version__ = "0+unknown"

from phiesta.remote.auth import connect_insula

from phiesta.l0 import L0_event, raw_l0_report, raw_l0_table
from phiesta.l1 import L1_event
from phiesta.l1a import L1A_event

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
    acquisition_report,
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
    "__version__",
    "connect_insula",
    "L0_event",
    "L1_event",
    "L1A_event",
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
    "acquisition_report",
    "interband_shift_table",
    "phisat2_band_table",
    "phisat2_product_level_specs",
    "mission_spec_report",
]
