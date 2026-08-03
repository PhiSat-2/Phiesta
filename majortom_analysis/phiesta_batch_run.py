import json
from pathlib import Path
from dotenv import load_dotenv
import os
from phiesta import connect_insula
import argparse
import pandas as pd
import shutil
   

def build_single_product_report(
            event,
            qc_report_path,
            georef_report_path: None
    ):
    
    with open(qc_report_path, "r") as qc_report:
            qc_report = json.load(qc_report)

    if georef_report_path.exists():
        with open(georef_report_path, "r") as georef_report:
            georef_report = json.load(georef_report)
    else:
        georef_report = {}

    qc_report_source = qc_report.get("sentinel_source", {})
    georef_report_metrics = georef_report.get("metrics", {})

    single_product_report = {
         # Product
         "phisat2_product_id": event._meta["catalog_geo"]["filename"], # productIdentifier

         # Sentinel source
         "sentinel_satellite": qc_report_source.get("satellite"),
         "num_tiles": qc_report_source.get("metadata", {}).get("num_tiles"),
         "sentinel2_products_name": [pair.get("l1c_name") for pair in qc_report_source.get("metadata", {}).get("pairs", [])],
         "delta_days": qc_report_source.get("delta_days"),
         "cloud_cover": qc_report_source.get("cloud_cover"),
         "coverage": qc_report_source.get("coverage"),         
        
         # Matching
         "raw_matches": georef_report_metrics.get("raw_matches"),
         "filtered_matches": georef_report_metrics.get("filtered_matches"),
         "inliers": georef_report_metrics.get("inliers"),
         "inlier_ratio": georef_report_metrics.get("inlier_ratio"),
         
         # Registration accuracy
         "error_mean_px": georef_report_metrics.get("error_mean_px"),
         "error_median_px": georef_report_metrics.get("error_median_px"),
         "error_p90_px": georef_report_metrics.get("error_p90_px"),
         "error_p95_px": georef_report_metrics.get("error_p95_px"),
         "error_max_px": georef_report_metrics.get("error_max_px"),
         
        }
    
    return single_product_report


def save_batch_run_configuration(
    full_triplet_report_path: str | Path,
    snr_psf_method: str,
    output_dir: str | Path = "majortom_analysis/"
):
    """
    Save the configuration used for a Phiesta batch run.
    """
    with open(full_triplet_report_path, "r") as full_triplet_report:
            full_triplet_report = json.load(full_triplet_report)

    full_triplet_report_config = full_triplet_report.get("config", {})

    config = {

        "window_days": full_triplet_report_config.get("window_days"),
        "max_cloud_cover": full_triplet_report_config.get("max_cloud_cover"),
        "buffer_km": full_triplet_report_config.get("buffer_km"),
        "snr_psf_method": snr_psf_method,
        "proxy_target_size": full_triplet_report_config.get("proxy_target_size"),
        "matching_max_side": full_triplet_report_config.get("matching_max_side"),
        "features": full_triplet_report_config.get("features"),
        "max_keypoints": full_triplet_report_config.get("max_keypoints"),
    }

    config_path = (Path(output_dir)/ "batch_config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    return config_path


def cleanup_phiesta_outputs(
    product_id: str,
    output_root: str | Path = "data/triplets",
    remove_triplet = False,
    remove_phisat_l1 = True,
    remove_sentinel_cache = True,
    remove_georef_file = False
):
    """
    Delete temporary files produced while processing one product.
    """

    output_root = Path(output_root)

    if remove_triplet:
        triplet_dir = output_root / product_id
        if triplet_dir.exists():
            shutil.rmtree(triplet_dir)

    if remove_phisat_l1:
        l1_dir = Path("data/l1")
        for folder in l1_dir.glob(f"*{product_id}*"):
            shutil.rmtree(folder, ignore_errors=True)

    if remove_sentinel_cache:
        cache_dir = Path("cache/sentinel2")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    if remove_georef_file:
        georef_dir = Path(f"georef_{product_id}.json")
        if georef_dir.exists():
            georef_dir.unlink()



def run_phiesta_over_one_product(
    product_id: str,
    snr_psf_method: str,
    output_root: str | Path = "data/triplets"
) -> dict:

    single_product_report = {
        # Product
        "phisat2_product_id": product_id,
        "status": None,
        "failure_type": None,
        "failure_reason": None,

        # Sentinel source
        "sentinel_satellite": None,
        "num_tiles": None,
        "sentinel2_products_name": None,
        "delta_days": None,
        "cloud_cover": None,
        "coverage": None,

        # Matching
        "raw_matches": None,
        "filtered_matches": None,
        "inliers": None,
        "inlier_ratio": None,

        # Registration accuracy
        "error_mean_px": None,
        "error_median_px": None,
        "error_p90_px": None,
        "error_p95_px": None,
        "error_max_px": None,
    }

    event = None
    georef_report_path = Path(f"georef_{product_id}.json")

    try:
        load_dotenv()

        client = connect_insula(
            username=os.getenv("INSULA_USERNAME"),
            password=os.getenv("INSULA_PASSWORD")
        )

        event = client.load_l1(product_id)

        georef = event.get_georef(
            sentinel_backend = "download",
            source = "simulated",
            snr_psf_method = snr_psf_method,
            cdse_username = os.getenv("CDSE_USERNAME"),
            cdse_password = os.getenv("CDSE_PASSWORD"),
            verbose = True,
        )

        georef_report_path.write_text(json.dumps(georef, indent=2), encoding="utf-8")

        single_product_report["status"] = "SUCCEEDED"       

    except Exception as e:
        failure_type = type(e).__name__
        failure_reason = str(e)

        single_product_report["status"] = "FAILED"
        single_product_report["failure_type"] = failure_type
        single_product_report["failure_reason"] = failure_reason

    try:
        if event is not None:
            qc_report_path = (Path(output_root)/event._meta["catalog_geo"]["identifier"]/ "qc.json")                    
            
            run_report = build_single_product_report(event, qc_report_path, georef_report_path)
            
            single_product_report.update(run_report)
            
    except Exception as e:
        print(f"Could not build report for {product_id}: {e}")

    return single_product_report



if __name__ == '__main__':
    # Reading CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', help='name of the file with the list of phisat2 productIdentifiers', required=True)
    args = parser.parse_args()

    config_saved = False
    reports_list = []

    snr_psf_method="alternative"
    output_root = "data/triplets"

    with open(args.txt) as phisat_pids:
        phisat_pids = [line.strip() for line in phisat_pids]

    for phisat_pid in phisat_pids:        
        single_product_report = run_phiesta_over_one_product(phisat_pid, snr_psf_method, output_root)

        full_triplet_report_path = (Path(output_root)/phisat_pid/ "full_triplet_report.json")

        if not config_saved and full_triplet_report_path.exists():
            save_batch_run_configuration(full_triplet_report_path, snr_psf_method)

            config_saved = True


        reports_list.append(single_product_report)

        cleanup_phiesta_outputs(
            phisat_pid,
            remove_triplet = True,
            remove_phisat_l1 = True,
            remove_sentinel_cache = True,
            remove_georef_file = True
        )

        pd.DataFrame(reports_list).to_csv(
            "majortom_analysis/phiesta_batch_run_results.csv",
            index=False,
        )
