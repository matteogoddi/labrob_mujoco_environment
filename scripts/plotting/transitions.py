"""Swing/stance transition detection, shared by contact_forces and joint_limits.

Extracted from the original contact-forces block: contact_forces.run()
detects transitions from per-corner Fz and returns them; joint_limits.run()
receives them explicitly to zoom in on the same time instants. This is the
one genuine cross-section data dependency in the whole diagnostic set — it
is now threaded through function arguments instead of living in shared
module/script scope.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Transition:
    idx: int
    kind: str  # 'touchdown' or 'liftoff'


def detect_stance_transitions(fz_cols: np.ndarray, stance_threshold_n: float,
                               min_gap_n: int) -> List[Transition]:
    """fz_cols: (n_rows, n_corners) per-corner normal force for one foot.

    A corner-force sum below stance_threshold_n is treated as "swing" (no
    contact). Re-crossings within min_gap_n samples of an accepted
    transition are ignored (debounce against contact-bounce chatter).
    """
    in_stance = fz_cols.sum(axis=1) > stance_threshold_n
    raw_edges = np.flatnonzero(np.diff(in_stance.astype(int)))
    edges = []
    for idx in raw_edges:
        if not edges or (idx - edges[-1]) > min_gap_n:
            edges.append(idx)
    return [Transition(idx=int(idx), kind='touchdown' if in_stance[idx + 1] else 'liftoff')
            for idx in edges]
