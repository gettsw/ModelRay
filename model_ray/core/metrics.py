"""Shared result schema, bootstrap CIs and the PASS/FAIL/INCONCLUSIVE rule."""

from __future__ import annotations

import math
from typing import Any, Callable, Literal, Sequence

import torch
from pydantic import BaseModel, Field, computed_field

Status = Literal["PASS", "FAIL", "INCONCLUSIVE", "INFO"]


class ProbeResult(BaseModel):
    """One gated metric. ``degradation`` is signed so that positive == worse."""

    probe: str
    metric: str
    unit: str  # e.g. "relative drop", "absolute increase"
    base_value: float
    tuned_value: float
    degradation: float
    ci_low: float
    ci_high: float
    threshold: float | None
    n: int
    issues: list[str] = Field(default_factory=list)  # validity problems => INCONCLUSIVE
    details: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> Status:
        """FAIL only if the whole CI is past the threshold, PASS only if the whole CI is
        within it; anything else (CI straddles, NaN, failed validity check) is INCONCLUSIVE."""
        if self.issues:
            return "INCONCLUSIVE"
        if self.threshold is None:
            return "INFO"
        if self.ci_low > self.threshold:
            return "FAIL"
        if self.ci_high <= self.threshold:
            return "PASS"
        return "INCONCLUSIVE"


def bootstrap_ci(
    stat: Callable[..., float],
    sizes: Sequence[int],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap. ``stat(*idx)`` takes one index tensor per group.

    Groups are resampled independently; pass the same indices to base and tuned
    inside ``stat`` to keep the comparison paired. Returns (point, low, high).
    NaN resamples (e.g. zero denominators) are ignored; all-NaN gives NaN bounds.
    """
    g = torch.Generator().manual_seed(seed)
    point = float(stat(*[torch.arange(n) for n in sizes]))
    boots = torch.tensor(
        [float(stat(*[torch.randint(n, (n,), generator=g) for n in sizes])) for _ in range(n_boot)]
    )
    if torch.isnan(boots).all():
        return point, math.nan, math.nan
    lo, hi = torch.nanquantile(boots, torch.tensor([alpha / 2, 1 - alpha / 2])).tolist()
    return point, lo, hi
