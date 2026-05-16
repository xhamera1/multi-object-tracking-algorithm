from typing import Iterable

from .types import TrackResult


def sort_results(results: Iterable[TrackResult]) -> list[TrackResult]:
    return sorted(results, key=lambda r: (r.frame, r.track_id))
