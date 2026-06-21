from enum import Enum

import geopandas as gpd

# Warning: Enums and %autoreload can clash, reset Jupyter kernel if you modify this file

# CONSTANTS
CROP_SIZE = (72, 72)
S2_RESOLUTION = 10
PHISAT2_RESOLUTION = 4.75
BBOX_SIZE = 20140  # in metres
BBOX_SIZE_CROPPED = 19456  # in metres
S2_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07"]
S2_PAN_BANDS = ["B02", "B03", "B04", "PAN", "B08", "B05", "B06", "B07"]
L1A_ABSOLUTE_SHIFTS = [6.329, 5.889, 5.339, 4.622, 3.785, 2.842, 1.796, 0]
L1A_RELATIVE_SHIFTS = [
    0,
    1.105,
    1.046,
    0.943,
    0.837,
    0.717,
    0.55,
    0.44,
]  # these values are in reverse order wrt the absolute shifts
L1A_RAND_STD = 0.4
L1A_RAND_MEAN = 0.0
PAN_WEIGHTS = [0.21594369, 0.28731533, 0.25719303, 0.0, 0.12275664, 0.11679131, 0.0]


class ProcessingLevels(Enum):
    L1A = "L1A"
    L1B = "L1B"
    L1C = "L1C"


from pathlib import Path


def _load_world_gdf():
    """
    Load a world countries GeoDataFrame if available.

    The original OrbitalAI simulation code used GeoPandas' built-in
    naturalearth_lowres dataset, but this was removed in GeoPandas >= 1.0.
    The simulation pipeline does not require this object for core S2 -> PhiSat-2
    simulation, so we fall back to an empty GeoDataFrame when unavailable.
    """
    try:
        return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        pass

    # Optional PyRawPh asset fallback if present.
    repo_root = Path(__file__).resolve().parents[2]
    asset_path = repo_root / "pyrawph" / "assets" / "world_countries.geojson"

    if asset_path.exists():
        try:
            return gpd.read_file(asset_path)
        except Exception:
            pass

    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


WORLD_GDF = _load_world_gdf()