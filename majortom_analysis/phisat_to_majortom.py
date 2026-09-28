from pathlib import Path
import pickle
import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.control import GroundControlPoint
import argparse
import ast
import json
import os
from shapely.geometry.base import BaseGeometry
from pyproj import Transformer
from tempfile import TemporaryDirectory

from majortom_grid_cell_retrival import phisat_intersection_with_majortom_grid, get_corrected_footprint
from phiesta_batch_run import run_phiesta_over_one_product, cleanup_phiesta_outputs
from majortom_upload import write_majortom_parquets, upload_majortom_parquets


def get_band_names(src_path):
    with rasterio.open(src_path) as src:
        return list(src.descriptions)


def make_json_serializable(obj):

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, BaseGeometry):
        return obj.__geo_interface__

    if isinstance(obj, dict):
        return {
            key: make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            make_json_serializable(value)
            for value in obj
        ]

    return obj
    

def build_corrected_phisat2_geotiff(
    georef_report,
    output_path,
    grid_size = 9,
):
    H_real_to_s2 = georef_report.get("H_real_to_s2")

    real_phisat_path = Path(georef_report["paths"]["real"])

    with rasterio.open(real_phisat_path) as real_phisat_src:
            real_phisat_w = real_phisat_src.width
            real_phisat_h = real_phisat_src.height
            real_phisat_profile = real_phisat_src.profile.copy()
            band_descriptions = real_phisat_src.descriptions
            dst_array = real_phisat_src.read()       

    xs = np.linspace(0, real_phisat_w - 1, grid_size)
    ys = np.linspace(0, real_phisat_h - 1, grid_size)

    xx, yy = np.meshgrid(xs, ys)

    real_phisat_points = np.column_stack([
        xx.ravel(),
        yy.ravel(),
    ])

    ones = np.ones(
        (len(real_phisat_points), 1),
        dtype=np.float64,
    )

    pts_h = np.concatenate(
        [real_phisat_points, ones],
        axis=1,
    )

    s2_h = (H_real_to_s2 @ pts_h.T).T

    s2_points = (
        s2_h[:, :2]
        / s2_h[:, 2:3]
    )

    final_sentinel_crop = Path(georef_report["paths"]["final_sentinel_crop"])

    with rasterio.open(final_sentinel_crop) as s2_src:
        s2_transform = s2_src.transform
        s2_crs = s2_src.crs

        map_x, map_y = s2_transform * (
            s2_points[:, 0],
            s2_points[:, 1],
        )

    transformer = Transformer.from_crs(
        s2_crs,
        "EPSG:4326",
        always_xy=True,
    )
    
    lon, lat = transformer.transform(
        map_x,
        map_y,
    )

    gcps = [
        GroundControlPoint(
            row = float(y),
            col = float(x),
            x = float(lon_i),
            y = float(lat_i),
            z = 0.0,
        )
        for x, y, lon_i, lat_i
        in zip(
            real_phisat_points[:, 0],
            real_phisat_points[:, 1],
            lon,
            lat,
        )
    ]

    real_phisat_profile.update(
        driver="GTiff",
        transform = None
    )

    with rasterio.open(output_path, "w", **real_phisat_profile,) as dst:
        dst.write(dst_array)
        dst.descriptions = band_descriptions
        dst.gcps = (
            gcps,
            rasterio.crs.CRS.from_epsg(4326),
        )


def extract_majortom_patch(
    raster_path: str | Path,
    majortom_cell,
) -> tuple[np.ndarray, np.ndarray]:

    target_crs = majortom_cell["utm_crs"]

    geotransform = ast.literal_eval(majortom_cell["geotransform"])

    target_transform = rasterio.Affine.from_gdal(*geotransform)

    target_height = 1056
    target_width = 1056

    with rasterio.open(raster_path) as src:

        src_gcps, src_gcp_crs = src.gcps

        patch = np.full(
            (
                src.count,
                target_height,
                target_width,
            ),
            65535,
            dtype=np.uint16,
        )

        reproject(
            source = src.read(),
            destination = patch,
            src_transform = None,
            src_crs = src_gcp_crs,
            gcps=src_gcps,

            dst_transform = target_transform,
            dst_crs = target_crs,
            dst_nodata = 65535,

            resampling = Resampling.nearest,
        )

    mask = (patch != 65535).all(axis=0).astype(np.uint8)

    return patch, mask


def build_majortom_sample(
    corrected_phisat2_path: str | Path,
    majortom_cell,
):
    """
    Build one Major-TOM triplet sample for one grid cell.

    """

    phisat2_real_patch, phisat2_real_mask = extract_majortom_patch(corrected_phisat2_path, majortom_cell)
    phisat2_real_bands = get_band_names(corrected_phisat2_path)
    
    row = {}

    for i, band_name in enumerate(phisat2_real_bands):
        row[f"real_phisat2_{band_name}"] = phisat2_real_patch[i]

    row["real_phisat2_mask"] = phisat2_real_mask

    return row


def build_majortom_metadata(
       single_product_report,
       majortom_cell,
       georef_report,
       phisat2_geometry
):
       
    single_batch_metadata = {

        "sample_id": (f"{single_product_report['phisat2_product_id']}_"f"{majortom_cell['grid_cell']}"),

        # Product
        "phisat2_product_id": single_product_report.get("phisat2_product_id"),
        "timestamp": single_product_report.get("timestamp"),
        "phisat2_corrected_geometry": phisat2_geometry,
        "phisat2_original_geometry": single_product_report.get("original_footprint"), 
        
        # Major-TOM
        "grid_cell": majortom_cell.get("grid_cell"),
        "grid_row_u": majortom_cell.get("grid_row_u"),
        "grid_col_r": majortom_cell.get("grid_col_r"),
        "geometry": majortom_cell.get("geometry"),
        "centre_lat": majortom_cell.get("centre_lat"),
        "centre_lon": majortom_cell.get("centre_lon"),
        "utm_footprint": majortom_cell.get("utm_footprint"),
        "utm_crs": majortom_cell.get("utm_crs"),
        "pixel_bbox": majortom_cell.get("pixel_bbox"),

        # Sentinel reference
        "sentinel_products": single_product_report.get("sentinel2_products_name"),
        "sentinel_satellite": single_product_report.get("sentinel_satellite"),
        "delta_days": single_product_report.get("delta_days"),
        "cloud_cover": single_product_report.get("cloud_cover"),
        "coverage": single_product_report.get("coverage"),
        "num_tiles": single_product_report.get("num_tiles"),

        # Homographies
        "H_real_to_s2" : georef_report.get("H_real_to_s2"),
        "H_s2_to_real" : georef_report.get("H_s2_to_real"),

        # Phiesta output quality
        "georef_error_mean_px": single_product_report.get("error_mean_px"),
        "georef_error_median_px": single_product_report.get("error_median_px"),
        "georef_error_p90_px": single_product_report.get("error_p90_px"),
        "georef_error_p95_px": single_product_report.get("error_p95_px"),
        "georef_error_max_px": single_product_report.get("error_max_px")

    }

    return single_batch_metadata    


def phisat_to_majortom_pipeline(
        georef_report_path: str | Path,
        single_product_report: dict,
    ) -> tuple[list[dict], list[dict]]:

    georef_report_path = Path(georef_report_path)

    with georef_report_path.open("r") as georef_report:
        georef_report = json.load(georef_report)

    phisat2_geometry = get_corrected_footprint(georef_report)
    majortom_cells = phisat_intersection_with_majortom_grid(georef_report)

    with TemporaryDirectory() as tmp_dir:
        corrected_phisat2_path = Path(tmp_dir) / "corrected_phisat2.tif"  
        build_corrected_phisat2_geotiff(georef_report, corrected_phisat2_path)        

        samples = []
        metadata = []

        for _, majortom_cell in majortom_cells.iterrows():
            sample = build_majortom_sample(corrected_phisat2_path, majortom_cell)

            sample_metadata = build_majortom_metadata(
                single_product_report = single_product_report,
                majortom_cell = majortom_cell,
                georef_report = georef_report,
                phisat2_geometry = phisat2_geometry
            )

            samples.append(sample)
            metadata.append(sample_metadata)

    return samples, metadata


if __name__ == '__main__':

    # Reading CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', help='name of the file with the list of phisat2 productIdentifiers', required=True)
    parser.add_argument('--tmp_dir')
  
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    snr_psf_method="alternative"    
    huggingface_repo_id = "Sara-inblue/Phisat_to_MajorTom"

    with open(args.txt) as phisat_pids:
        phisat_pids = [line.strip() for line in phisat_pids]

    for phisat_pid in phisat_pids:        
        single_product_report = run_phiesta_over_one_product(phisat_pid, snr_psf_method, tmp_dir)

        if single_product_report["status"] != "SUCCEEDED":
            continue

        georef_report_path = tmp_dir / phisat_pid / f"georef_{phisat_pid}.json"

        try:

            majortom_samples, majortom_metadata = phisat_to_majortom_pipeline(georef_report_path, single_product_report)

            samples_path = (tmp_dir / f"{phisat_pid}_samples.pkl")
            metadata_path = (tmp_dir / f"{phisat_pid}_metadata.pkl")

            with samples_path.open("wb") as f:
                pickle.dump(majortom_samples, f)

            with metadata_path.open("wb") as f:
                pickle.dump(majortom_metadata, f)

            write_majortom_parquets(tmp_dir, phisat_pid, huggingface_repo_id)

            upload_majortom_parquets(tmp_dir, phisat_pid, huggingface_repo_id)

        except Exception as e:

            print(
                f"Major-TOM conversion failed for "
                f"{phisat_pid}: {e}"
            )

        cleanup_phiesta_outputs(
            phisat_pid,
            tmp_dir,
            remove_triplet = True,
            remove_phisat_l1 = True,
            remove_sentinel_cache = True,
            remove_georef_file = True
        )
