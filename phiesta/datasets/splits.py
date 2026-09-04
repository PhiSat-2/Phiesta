from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
_SPLITS = ("train", "val", "test")


def _validate_ratios(train, val, test):
    ratios = {"train": float(train), "val": float(val), "test": float(test)}
    if any(v < 0 for v in ratios.values()):
        raise ValueError("train, val, and test ratios must be >= 0.")
    total = sum(ratios.values())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"train + val + test must equal 1.0, got {total:.12g}.")
    return ratios


def _group_fields(group_by, product_id_col, columns):
    if group_by is None:
        fields = [product_id_col]
    elif isinstance(group_by, str):
        fields = [product_id_col] if group_by.lower() in {
            "acquisition", "acquisitions", "product", "product_id"
        } else [group_by]
    else:
        fields = list(group_by)
    missing = [c for c in fields if c not in columns]
    if missing:
        raise ValueError(f"Missing group_by column(s): {missing}.")
    return fields


def _keys(df, fields):
    values = df[fields].where(df[fields].notna(), "<NA>")
    return [
        tuple(str(v) for v in row)
        for row in values.itertuples(index=False, name=None)
    ]


def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class _UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _random_units(df, fields):
    groups = {}
    for i, key in enumerate(_keys(df, fields)):
        groups.setdefault(key, []).append(i)
    return list(groups.values())


def _spatial_units(df, fields, min_distance_km, lon_col, lat_col):
    if min_distance_km is None or float(min_distance_km) <= 0:
        raise ValueError("method='spatial' requires min_distance_km > 0.")

    for col in (lon_col, lat_col):
        if col not in df.columns:
            raise ValueError(f"Spatial splitting requires column {col!r}.")

    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
    bad = ~(np.isfinite(lon) & np.isfinite(lat))
    if bad.any():
        ids = df.loc[bad, "product_id"].astype(str).tolist()[:10]
        raise ValueError(f"Missing finite catalog center coordinates for: {ids}")

    uf = _UF(len(df))

    # Respect arbitrary grouping first.
    first = {}
    for i, key in enumerate(_keys(df, fields)):
        if key in first:
            uf.union(first[key], i)
        else:
            first[key] = i

    # Then connect all acquisitions closer than the requested center distance.
    # Connected components become indivisible split units.
    for i in range(len(df) - 1):
        distances = _haversine_km(
            lon[i], lat[i], lon[i + 1 :], lat[i + 1 :]
        )
        for offset in np.flatnonzero(distances < float(min_distance_km)):
            uf.union(i, i + 1 + int(offset))

    components = {}
    for i in range(len(df)):
        components.setdefault(uf.find(i), []).append(i)
    return list(components.values())


def _assign(units, ratios, seed):
    active = [name for name in _SPLITS if ratios[name] > 0]
    total = float(sum(len(unit) for unit in units))
    targets = {name: ratios[name] * total for name in active}
    counts = {name: 0.0 for name in active}

    rng = np.random.default_rng(int(seed))
    noise = rng.random(len(units))
    order = sorted(
        range(len(units)),
        key=lambda i: (-len(units[i]), float(noise[i])),
    )
    out = [""] * len(units)

    for i in order:
        weight = float(len(units[i]))
        choices = []
        for split in active:
            trial = dict(counts)
            trial[split] += weight
            score = sum(
                ((trial[name] - targets[name]) ** 2) / max(targets[name], 1.0)
                for name in active
            )
            choices.append((score, split))

        best = min(score for score, _ in choices)
        tied = [
            split
            for score, split in choices
            if np.isclose(score, best, rtol=0.0, atol=1e-12)
        ]
        chosen = tied[int(rng.integers(0, len(tied)))]
        out[i] = chosen
        counts[chosen] += weight

    return out


def split_summary(dataset, split_col="split"):
    acq = dataset.acquisitions
    patches = dataset.patches
    if split_col not in acq.columns:
        return pd.DataFrame(
            columns=["split", "acquisitions", "patches", "acquisition_fraction"]
        )

    assigned = acq[acq[split_col].notna()].copy()
    total = len(assigned)
    present = set(assigned[split_col].astype(str))
    names = [s for s in _SPLITS if s in present] + sorted(present - set(_SPLITS))

    rows = []
    for name in names:
        n_acq = int((assigned[split_col].astype(str) == name).sum())
        n_patch = 0
        if not patches.empty and split_col in patches.columns:
            n_patch = int((patches[split_col].astype(str) == name).sum())
        rows.append({
            "split": name,
            "acquisitions": n_acq,
            "patches": n_patch,
            "acquisition_fraction": n_acq / total if total else 0.0,
        })
    return pd.DataFrame(rows)


def get_split(dataset, name, level="auto", split_col="split"):
    level = str(level).lower()
    if level == "auto":
        level = "patches" if not dataset.patches.empty else "acquisitions"
    if level == "patches":
        table = dataset.patches
    elif level == "acquisitions":
        table = dataset.acquisitions
    else:
        raise ValueError("level must be 'auto', 'patches', or 'acquisitions'.")

    if split_col not in table.columns:
        raise ValueError("Dataset has no split assignments. Run make_splits() first.")

    return (
        table[table[split_col].astype(str) == str(name)]
        .copy()
        .reset_index(drop=True)
    )


def make_splits(
    dataset,
    train=0.8,
    val=0.1,
    test=0.1,
    method="random",
    group_by=None,
    seed=42,
    min_distance_km=None,
    lon_col="center_lon",
    lat_col="center_lat",
    product_id_col="product_id",
    split_col="split",
    include_failed=False,
    overwrite=False,
    verbose=True,
):
    """
    Create acquisition/group-level train/val/test splits and propagate them
    to every patch.

    ``method='random'`` supports arbitrary ``group_by`` manifest columns.
    ``method='spatial'`` additionally forces acquisitions whose catalog centers
    are closer than ``min_distance_km`` into one connected component. Therefore
    acquisitions in different splits have at least that center separation.
    """
    ratios = _validate_ratios(train, val, test)
    acq = dataset.acquisitions.copy()
    patches = dataset.patches.copy()

    if acq.empty:
        raise ValueError("Dataset has no acquisitions to split.")
    if product_id_col not in acq.columns:
        raise ValueError(f"Missing {product_id_col!r} in acquisitions manifest.")
    if split_col in acq.columns and acq[split_col].notna().any() and not overwrite:
        raise ValueError("Split assignments already exist. Use overwrite=True to replace them.")

    eligible_mask = pd.Series(True, index=acq.index)
    if not include_failed and "build_status" in acq.columns:
        eligible_mask = acq["build_status"].astype(str) == "SUCCESS"
    eligible = acq.loc[eligible_mask].copy()
    if eligible.empty:
        raise ValueError("No eligible acquisitions are available for splitting.")

    fields = _group_fields(group_by, product_id_col, eligible.columns)
    method = str(method).lower().strip()
    if method in {"random", "group", "grouped"}:
        method = "random"
        units = _random_units(eligible, fields)
        effective_distance = None
    elif method == "spatial":
        units = _spatial_units(
            eligible,
            fields,
            min_distance_km,
            lon_col,
            lat_col,
        )
        effective_distance = float(min_distance_km)
    else:
        raise ValueError("method must be 'random' or 'spatial'.")

    assignments = _assign(units, ratios, seed)

    # overwrite=True explicitly replaces all previous assignments; otherwise
    # there were no non-null assignments because of the guard above.
    acq[split_col] = pd.NA
    acq["split_group_id"] = pd.NA
    original_indices = list(eligible.index)

    for unit_id, (positions, split) in enumerate(zip(units, assignments)):
        indices = [original_indices[pos] for pos in positions]
        acq.loc[indices, split_col] = split
        acq.loc[indices, "split_group_id"] = f"{method}_{unit_id:05d}"

    # Propagate only from acquisition assignments. Patches from one acquisition
    # can therefore never leak across splits.
    if not patches.empty:
        if product_id_col not in patches.columns:
            raise ValueError(f"Missing {product_id_col!r} in patch manifest.")
        mapping = acq[[product_id_col, split_col, "split_group_id"]].copy()
        mapping[product_id_col] = mapping[product_id_col].astype(str)
        split_map = mapping.set_index(product_id_col)[split_col].to_dict()
        group_map = mapping.set_index(product_id_col)["split_group_id"].to_dict()
        patches[split_col] = patches[product_id_col].astype(str).map(split_map)
        patches["split_group_id"] = patches[product_id_col].astype(str).map(group_map)

    dataset.acquisitions = acq.reset_index(drop=True)
    dataset.patches = patches.reset_index(drop=True)

    manifest_cols = [product_id_col, split_col, "split_group_id"]
    for col in fields + [lon_col, lat_col]:
        if col in dataset.acquisitions.columns and col not in manifest_cols:
            manifest_cols.append(col)
    dataset.acquisitions.loc[
        dataset.acquisitions[split_col].notna(), manifest_cols
    ].to_csv(dataset.root / "splits.csv", index=False)

    summary = split_summary(dataset, split_col=split_col)
    dataset.metadata["splits"] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "train": ratios["train"],
        "val": ratios["val"],
        "test": ratios["test"],
        "seed": int(seed),
        "group_by": fields,
        "min_distance_km": effective_distance,
        "eligible_acquisitions": int(len(eligible)),
        "split_units": int(len(units)),
        "counts": summary.to_dict(orient="records"),
    }

    from .builder import _write_state
    _write_state(dataset.root, dataset.acquisitions, dataset.patches, dataset.metadata)

    if verbose:
        print(
            f"[Phiesta splits] method={method}, "
            f"acquisitions={len(eligible)}, groups={len(units)}"
        )
        if not summary.empty:
            print(summary.to_string(index=False))
    return summary
