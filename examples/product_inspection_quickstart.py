from pathlib import Path
import json

import phiesta


OUT = Path("outputs/examples/product_inspection_quickstart")
OUT.mkdir(parents=True, exist_ok=True)


def make_json_safe(obj):
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return obj


def save_json(obj, path):
    path.write_text(json.dumps(make_json_safe(obj), indent=2, default=str))
    print("wrote", path)


def main():
    # 1. Open one local or remote PhiSat-2 product.
    l1c = phiesta.open_product("6008", level="L1C")

    # 2. Inspect product content.
    card = phiesta.product_card(l1c)
    manifest = phiesta.file_manifest(l1c)
    families = phiesta.file_family_summary(l1c)
    switches = phiesta.processing_switches(l1c)
    rasters = phiesta.raster_inventory(l1c)

    save_json(card, OUT / "6008_l1c_product_card.json")
    manifest.to_csv(OUT / "6008_l1c_file_manifest.csv", index=False)
    families.to_csv(OUT / "6008_l1c_file_family_summary.csv", index=False)
    switches.to_csv(OUT / "6008_l1c_processing_switches.csv", index=False)
    rasters.to_csv(OUT / "6008_l1c_raster_inventory.csv", index=False)

    print("\n=== Product card ===")
    print(json.dumps(make_json_safe(card), indent=2, default=str))

    # 3. Compute a heuristic visual/product screening report.
    quality = phiesta.quality_report(l1c)
    save_json(quality, OUT / "6008_l1c_quality_report.json")

    print("\n=== Quality report ===")
    print(json.dumps(make_json_safe(quality), indent=2, default=str))

    # 3b. Compare the product against mission-level expectations.
    mission_report = phiesta.mission_spec_report(l1c)
    save_json(mission_report, OUT / "6008_l1c_mission_spec_report.json")

    print("\n=== Mission spec report ===")
    print(json.dumps(make_json_safe({
        "product_id": mission_report["product_id"],
        "level": mission_report["level"],
        "overall_ok": mission_report["overall_ok"],
        "checks": mission_report["checks"],
    }), indent=2, default=str))

    # 4. Build a screening gallery from product ids.
    product_ids = ["5978", "5979", "5980", "5987", "6008", "6018", "6025", "6038", "6040", "6041", "6045"]
    gallery_table = phiesta.product_gallery(
        product_ids,
        level="L1C",
        out_path=OUT / "l1c_screening_gallery.png",
        title="PhiSat-2 L1C screening gallery",
        ncols=4,
    )
    gallery_table.to_csv(OUT / "l1c_screening_gallery_table.csv", index=False)

    # 5. Compare two processing levels from the same acquisition when both are available.
    try:
        l1a = phiesta.open_product("6008", level="L1A")
        comparison = phiesta.compare_levels(
            l1a,
            l1c,
            include_shift=True,
            include_mission_specs=True,
            master_band=2,
        )
        save_json(comparison, OUT / "6008_l1a_l1c_comparison.json")

        print("\n=== L1A/L1C inter-band shift summary ===")
        print(json.dumps(make_json_safe(comparison["interband_shift"]), indent=2, default=str))

        shifts_l1a = phiesta.interband_shift_table(l1a, master_band=2)
        shifts_l1c = phiesta.interband_shift_table(l1c, master_band=2)
        shifts_l1a.to_csv(OUT / "6008_l1a_interband_shifts.csv", index=False)
        shifts_l1c.to_csv(OUT / "6008_l1c_interband_shifts.csv", index=False)

    except Exception as e:
        print("Skipping L1A/L1C comparison:", type(e).__name__, e)

    print("\nDone. Outputs written to:", OUT)


if __name__ == "__main__":
    main()
