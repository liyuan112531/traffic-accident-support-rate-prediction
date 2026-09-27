"""Item-indicator block.

The meta-learner sees case-item units pooled across items, so it cannot infer
which item a row belongs to from the request-side features alone unless the
item is encoded. A one-hot indicator of the 16 items supplies that context::

    16 item indicators

Together with the 33 request features these form the 49-dimensional claim-side
block of the meta representation (``33 + 16 = 49``). The indicators are known
before judgment: they record which head of claim a unit belongs to, not how much
was awarded.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..preprocessing.schema import ITEMS, N_ITEMS

#: Width of the indicator block.
N_ITEM_INDICATORS: int = N_ITEMS


def item_indicator_names(items: Sequence[str] = ITEMS) -> List[str]:
    """Ordered names of the item-indicator columns.

    The identifier is ``item_<item>``, matching the naming used by the study's
    meta feature matrix and by the feature labels in the manuscript.
    """
    return [f"item_{i}" for i in items]


def item_indicator_row(item: str, items: Sequence[str] = ITEMS) -> np.ndarray:
    """One-hot row for a single item; raises on an unknown identifier."""
    if item not in items:
        raise KeyError(f"unknown item identifier: {item!r}")
    row = np.zeros(len(items), dtype=float)
    row[list(items).index(item)] = 1.0
    return row


def item_indicator_matrix(item_values: Sequence[str],
                          items: Sequence[str] = ITEMS) -> np.ndarray:
    """One-hot matrix for a sequence of item identifiers, shape ``(n, 16)``."""
    lookup = {name: j for j, name in enumerate(items)}
    out = np.zeros((len(item_values), len(items)), dtype=float)
    for row, name in enumerate(item_values):
        try:
            out[row, lookup[str(name)]] = 1.0
        except KeyError as exc:
            raise KeyError(f"unknown item identifier: {name!r}") from exc
    return out
