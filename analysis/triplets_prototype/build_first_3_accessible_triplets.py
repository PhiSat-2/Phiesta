from pathlib import Path
import json
import traceback

from pyrawph import connect_insula

N_WANTED = 3
MAX_PRODUCTS_TO_TRY = 30

client = connect_insula()

df = client.search_l1_table(
    page=0,
    results_per_page=MAX_PRODUCTS_TO_TRY,
)

print("Candidate products:")
print(df[["product_id", "filename", "start_datetime", "center_lon", "center_lat"]].head(MAX_PRODUCTS_TO_TRY).to_string(index=False))

rows = []
success_count = 0

for _, row in df.iterrows():
    product_id = str(row["product_id"])

    # Skip 5359: already done
    if product_id == "5359":
        continue

    print("\n" + "=" * 100)
    print(f"TRY FULL TRIPLET — {product_id}")
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

        success_count += 1
        print(f"[SUCCESS] {product_id} ({success_count}/{N_WANTED})")

        if success_count >= N_WANTED:
            break

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

out_json = out_dir / "triplet_build_first_3_accessible_summary.json"
out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
for r in rows:
    print(r)

print("\nSaved:", out_json)
