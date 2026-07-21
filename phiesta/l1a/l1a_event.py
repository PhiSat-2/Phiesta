from __future__ import annotations

from phiesta.l1.l1_event import L1_event


class L1A_event(L1_event):
    """
    Provisional ΦSat-2 L1A event wrapper.

    This intentionally reuses the current L1/L1C loader machinery until the
    exact L1A archive structure is inspected. The goal is first to download,
    open, and audit L1A products, not to assume full L1C compatibility.
    """

    processing_level = "L1A"
