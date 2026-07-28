from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import math

import matplotlib.pyplot as plt
import pandas as pd

from .anatomy import _as_path
from .opening import open_product
from .quality import quality_report, _preferred_display_raster, _read_display_sample


def _resolve_product(product: Any, *, level: str = "L1C", prefer_local: bool = True):
    if isinstance(product, (str, int, Path)):
        return open_product(product, level=level, prefer_local=prefer_local)
    return product


def product_gallery(
    products: Iterable[Any],
    *,
    level: str = "L1C",
    out_path: str | Path | None = None,
    title: str | None = None,
    max_side: int = 512,
    ncols: int = 4,
    sort_by: str | None = None,
    prefer_local: bool = True,
) -> pd.DataFrame:
    """
    Render a visual gallery of PhiSat-2 products with heuristic screening scores.

    Parameters
    ----------
    products:
        Product ids, local folders, or already loaded Phiesta events.
    level:
        Product level used when products are ids or paths.
    out_path:
        Optional PNG output path.
    title:
        Optional figure title.
    max_side:
        Maximum side of each displayed thumbnail sample.
    ncols:
        Number of columns in the gallery.
    sort_by:
        Optional column used to sort the gallery, e.g. "score".
    prefer_local:
        Prefer local product folders before remote Insula loading.

    Returns
    -------
    pandas.DataFrame
        Quality/screening table used to annotate the gallery.
    """
    resolved = []
    rows = []

    for product in products:
        event = _resolve_product(product, level=level, prefer_local=prefer_local)
        report = quality_report(event, max_side=max_side)
        resolved.append(event)
        rows.append(report)

    df = pd.DataFrame(rows)

    order = list(range(len(resolved)))
    if sort_by is not None and sort_by in df.columns:
        ascending = sort_by != "score"
        order = df.sort_values(sort_by, ascending=ascending).index.tolist()

    n = len(order)
    if n == 0:
        raise ValueError("No products to display.")

    ncols = max(1, int(ncols))
    nrows = math.ceil(n / ncols)

    fig_w = 4.2 * ncols
    fig_h = 4.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for ax_i, row_i in enumerate(order):
        ax = axes.ravel()[ax_i]
        event = resolved[row_i]
        report = rows[row_i]

        root = _as_path(event)
        raster = _preferred_display_raster(root)

        ax.axis("off")

        if raster is None:
            ax.text(0.5, 0.5, "missing raster", ha="center", va="center")
            continue

        try:
            rgb, _meta = _read_display_sample(raster, max_side=max_side)
            ax.imshow(rgb)
        except Exception as e:
            ax.text(0.5, 0.5, f"read failed\n{type(e).__name__}", ha="center", va="center")

        flags = report.get("flags", [])
        if isinstance(flags, list):
            flags_txt = ", ".join(flags[:2])
            if len(flags) > 2:
                flags_txt += ", ..."
        else:
            flags_txt = str(flags)

        label = (
            f"{report.get('product_id')}  {report.get('level')}\n"
            f"score={report.get('score')}  {report.get('recommendation')}"
        )
        if flags_txt:
            label += f"\n{flags_txt}"

        ax.set_title(label, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=170)
        print(f"wrote {out}")

    plt.close(fig)

    return df
