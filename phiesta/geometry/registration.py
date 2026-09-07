from __future__ import annotations

from typing import Any

from phiesta.utils.l0_l1_registration import (
    register_event_bands_to_master,
    register_l0_to_l1_space,
)


def _resolve_registration_band(event: Any, band: Any) -> int:
    if isinstance(band, int):
        return int(band)

    resolver = getattr(event, "_resolve_band_index_for_registration", None)
    if callable(resolver):
        return int(resolver(band))

    raise ValueError(
        f"Cannot resolve band {band!r} for this event. "
        "Pass an integer band index."
    )


def register_bands(
    event: Any,
    *,
    master_band: Any = "RED",
    max_shifts=(80, 80),
):
    """
    Return a copy of one event with all bands translated into a master-band space.

    This exposes Phiesta's existing intra-event registration utility as a public
    analysis API. The original event is not modified.
    """
    master_idx = _resolve_registration_band(event, master_band)
    return register_event_bands_to_master(
        event,
        master_band=master_idx,
        max_shifts=max_shifts,
    )


def register_l0_to_l1(
    l0_event: Any,
    l1_event: Any,
    *,
    master_band: Any = "NIR",
    max_shifts=(300, 300),
):
    """
    Register an L0 event into the reference space of a paired L1 event.

    The returned event records the master L0→L1 translation, per-band residual
    translations, crop strategy, and target L1 geospatial metadata in
    ``registration_info``.
    """
    master_idx = _resolve_registration_band(l0_event, master_band)
    return register_l0_to_l1_space(
        l0_event,
        l1_event,
        master_band=master_idx,
        max_shifts=max_shifts,
    )
