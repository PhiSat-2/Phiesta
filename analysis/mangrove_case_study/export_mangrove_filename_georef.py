from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from pyrawph import connect_insula
from pyrawph.remote.constants import PHISAT2_L1_COLLECTION
from pyrawph.remote.catalog_geometry import catalog_geo_from_feature


SRC = Path("outputs/mangrove_candidates_cloud_scored.csv")
OUT = Path("outputs/mangrove_candidates_filename_georef.csv")


def extract_product_id(text):
    if text is None:
        return None

    s = str(text)

    m = re.search(r"PHISAT-2_L1_0*(\d+)_", s)
    if m:
        return str(int(m.group(1)))

    m = re.search(r"\b0*(\d{3,8})\b", s)
    if m:
        return str(int(m.group(1)))

    return None


def collect_strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from collect_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from collect_strings(v)
    elif isinstance(obj, str):
        yield obj


def extract_full_filename(feature):
    strings = list(collect_strings(feature))

    # Best case: full ΦSat-2 L1 product name.
    for s in strings:
        m = re.search(r"PHISAT-2_L1_[A-Za-z0-9_]+", s)
        if m:
            return m.group(0)

    # Fallback: any string containing PHISAT-2_L1.
    for s in strings:
        if "PHISAT-2_L1" in s:
            return s

    return ""


def main():
    df = pd.read_csv(SRC)

    review = df[
        (df["cloud_eval_status"] == "SUCCESS")
        & (df["cloud_status"] == "low_cloud")
    ].copy()

    review = review.sort_values(
        ["mangrove_pixels", "mangrove_fraction"],
        ascending=False,
    )

    target_ids = set(review["product_id"].astype(int).astype(str))

    print("Targets:", len(target_ids))

    client = connect_insula()

    records = {}
    page = 0

    while target_ids - set(records.keys()):
        print(f"[Insula] page {page}")

        data = client.search_ref_data(
            ref_data_collection=PHISAT2_L1_COLLECTION,
            page=page,
            results_per_page=100,
        )

        features = data.get("features", [])
        print("features:", len(features))

        if not features:
            break

        for feature in features:
            full_filename = extract_full_filename(feature)

            pid = (
                extract_product_id(full_filename)
                or extract_product_id(feature.get("id"))
                or extract_product_id(json.dumps(feature)[:5000])
            )

            if pid not in target_ids:
                continue

            catalog_geo = catalog_geo_from_feature(feature)

            records[pid] = {
                "full_filename": full_filename,
                "insula_georef": json.dumps(
                    catalog_geo,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }

            print("found", pid, full_filename)

        page += 1

    rows = []

    for _, row in review.iterrows():
        pid = str(int(row["product_id"]))
        rec = records.get(pid)

        if rec is None:
            # Fallback from CSV bbox if full Insula feature was not recovered.
            georef = {
                "type": "bbox_lonlat",
                "bbox": [
                    float(row["bbox_min_lon"]),
                    float(row["bbox_min_lat"]),
                    float(row["bbox_max_lon"]),
                    float(row["bbox_max_lat"]),
                ],
                "center": [
                    float(row["center_lon"]),
                    float(row["center_lat"]),
                ],
            }

            rows.append({
                "full_filename": f"PHISAT-2_L1_00000{pid}",
                "insula_georef": json.dumps(
                    georef,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            })
        else:
            rows.append(rec)

    out = pd.DataFrame(rows, columns=["full_filename", "insula_georef"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print()
    print("saved:", OUT)
    print("rows:", len(out))
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
