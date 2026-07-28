from pathlib import Path
import csv
import math

from phiesta.l1a import L1A_event
from phiesta.l1 import L1_event

PRODUCT_IDS = ["5978", "5979", "5987", "6008", "6018", "6025", "6038", "6040", "6041", "6045"]
OUT = Path("outputs/l1a_l1c_audit_selected/interband_shifts.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = []

def load_event(root, level, pid, cls, product_kind="BC"):
    p = sorted(Path(root).glob(f"PHISAT-2_{level}_{int(pid):09d}_*"))[0]
    return cls.from_path(
        p,
        product_kind=product_kind,
        multiband=True,
        as_float32=False,
        verbose=False,
    )

for pid in PRODUCT_IDS:
    for level, root, cls in [
        ("L1A", "data/l1a", L1A_event),
        ("L1C", "data/l1", L1_event),
    ]:
        ev = load_event(root, "L1A" if level == "L1A" else "L1", pid, cls)

        for master in [2, 7]:
            reg = ev._registered_for_display(
                master_band=master,
                max_shifts=(120, 120),
                force=True,
            )

            info = reg.meta["band_registration_info"]
            shifts = info["band_shifts_to_master_dy_dx"]

            for band_str, shift in sorted(shifts.items(), key=lambda kv: int(kv[0])):
                band = int(band_str)
                dy, dx = float(shift[0]), float(shift[1])
                rows.append({
                    "product_id": pid,
                    "level": level,
                    "product_kind": "BC",
                    "master_band": master,
                    "band": band,
                    "dy": dy,
                    "dx": dx,
                    "shift_mag": math.sqrt(dy * dy + dx * dx),
                })

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("wrote", OUT)
