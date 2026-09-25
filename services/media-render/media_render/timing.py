"""Where the words land in a spoken line: energy-gap detection (PRD-251A stage 2).

The reference timed on-screen text to the voice this way: the line's energy in
10 ms hops, silent where it falls under 4% of the line's peak, and a gap
wherever the silence lasts 70 ms or more. Each voiced run between gaps is a
segment: a word, or a phrase spoken without a pause. That is how "Orders.
Stock. Customers. The books." moved a board card on each word.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

HOP_SECONDS = 0.010
SILENCE_FRACTION = 0.04
MIN_GAP_SECONDS = 0.070


def voiced_segments(samples: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
    """(start, end) seconds of each voiced run, split at gaps of MIN_GAP_SECONDS or more."""
    mono = np.asarray(samples, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    hop = max(1, int(round(sample_rate * HOP_SECONDS)))
    hops = -(-mono.size // hop)
    if hops == 0:
        return []
    padded = np.zeros(hops * hop)
    padded[: mono.size] = mono
    energy = np.sqrt(np.mean(padded.reshape(hops, hop) ** 2, axis=1))
    peak = float(energy.max())
    if peak <= 0.0:
        return []
    voiced = np.flatnonzero(energy >= SILENCE_FRACTION * peak)
    hop_seconds = hop / sample_rate
    min_gap_hops = int(round(MIN_GAP_SECONDS / hop_seconds))
    runs: List[Tuple[int, int]] = []
    start = previous = int(voiced[0])
    for index in (int(i) for i in voiced[1:]):
        if index - previous - 1 >= min_gap_hops:
            runs.append((start, previous + 1))
            start = index
        previous = index
    runs.append((start, previous + 1))
    duration = mono.size / sample_rate
    return [(round(first * hop_seconds, 3), round(min(last * hop_seconds, duration), 3)) for first, last in runs]
