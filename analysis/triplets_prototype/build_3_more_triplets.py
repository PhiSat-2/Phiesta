from pathlib import Path
import json
import traceback

from pyrawph import connect_insula

PRODUCT_IDS = ["3702", "2966", "2400"]

client = connect_insula()

rows = []

for product_id in PRODUCT_IDS:
    print("\n" + "=" * 100)
    print(f"BUILD FULL TRIPLET — {product_id}")
    print("=" * 100)

    try:
        event = client.load_l1(product_id)

        triplet = event.build_full_sentinel_triplet(
            buffer_km=20.0,
            proxy_target_size=(1024, 1024),
            final_margin_pct=0.15,
            final_simulation_target_size=None,
            verbose=True,
        )

        event.inspect_full_sentinel_triplet(triplet)

        paths = triplet.get("paths", triplet)

        rows.append({
            "product_id": product_id,
            "status": "SUCCESS",
            "real": paths.get("real"),
            "sentinel": paths.get("sentinel"),
            "simulated": paths.get("simulated"),
            "metadata": paths.get("metadata"),
            "report": paths.get("report"),
        })

    except Exception as exc:
        print(f"[FAILED] {product_id}: {type(exc).__name__}: {exc}")
        traceback.print_exc()

        rows.append({
            "product_id": product_id,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })

out_dir = Path("outputs/triplet_build_test")
out_dir.mkdir(parents=True, exist_ok=True)

out_json = out_dir / "triplet_build_3_more_summary.json"
out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
for r in rows:
    print(r)

print("\nSaved:", out_json)
