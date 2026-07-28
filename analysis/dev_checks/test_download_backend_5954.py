from pyrawph import connect_insula

PRODUCT_ID = "5954"

print("Step 1/3 — Connect to Insula")
client = connect_insula()

print("Step 2/3 — Load PhiSat-2 L1 product from Insula/local cache")
event = client.load_l1(PRODUCT_ID)

print("Step 3/3 — Build triplet with portable Sentinel download backend")
triplet = event.build_full_sentinel_triplet(
    buffer_km=20.0,
    proxy_target_size=(1024, 1024),
    final_margin_pct=0.15,
    final_simulation_target_size=None,
    sentinel_backend="download",
    sentinel_cache_dir="cache/sentinel2",
    verbose=True,
)

event.inspect_full_sentinel_triplet(triplet)

print("\nDONE")
print(triplet)
