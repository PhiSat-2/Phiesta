import json
import geopandas as gpd
from huggingface_hub import hf_hub_download
from pathlib import Path

from shapely.geometry import shape


def get_corrected_footprint(
    georef_report_path: str | Path,
) -> list[str]:
    """
    Read the corrected PhiSat-2 footprint from Phiesta report
    
    """
    with open(georef_report_path, "r") as f:
        georef_report = json.load(f)

    footprint_geojson = georef_report["polygon_geojson"]

    footprint = shape(footprint_geojson)

    footprint_gdf = gpd.GeoDataFrame(
        {"geometry": [footprint]},
        crs="EPSG:4326",
    )

    return footprint_gdf.geometry.iloc[0]


def phisat_intersection_with_majortom_grid(
    georef_report_path: str | Path,
):

    # get MajorTom grid
    grid_path = hf_hub_download(
            repo_id="Major-TOM/Spatial-Reference-Grid",
            filename="global/MT_grid_10km_global.parquet",
            repo_type="dataset",
        )
    
    grid = gpd.read_parquet(grid_path)

    # get Phisat2 footprint corrected using phiesta
    phisat_footprint = get_corrected_footprint(georef_report_path)

    intersecting_tiles = grid[
        grid.intersects(phisat_footprint)
    ].copy()

    return intersecting_tiles




if __name__ == '__main__':
    
    tiles = phisat_intersection_with_majortom_grid(
        "/home/sromagnoli/Phiesta/georef_6039.json"
    )

    print(tiles)

