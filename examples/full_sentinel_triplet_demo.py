from pyrawph import connect_insula

client = connect_insula()

# Load one PhiSat-2 L1 acquisition
event = client.load_l1("5359")

# Build a full pixel-aligned triplet:
# real PhiSat-2, Sentinel-2B, simulated PhiSat-2
triplet = event.build_full_sentinel_triplet(
    buffer_km=20.0,
    proxy_target_size=(1024, 1024),
    final_margin_pct=0.15,
    verbose=True,
)

# Inspect outputs
event.inspect_full_sentinel_triplet(triplet)

# Display RGB comparison and PAN overlay
event.show_full_sentinel_triplet(
    triplet,
    save_dir="data/triplets/5359/quicklooks",
)

print(triplet["paths"])
