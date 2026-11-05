"""Safety probes. Each has ``measure(auditor) -> tensors`` (one model at a time, so
base and tuned never need to be in memory together) and
``compare(base, tuned) -> list[ProbeResult]``."""

from model_ray.probes.eval_awareness import ContextItem, EvaluationAwarenessProbe
from model_ray.probes.overwrite import OverwriteDistanceProbe, SycophancyItem
from model_ray.probes.refusal import RefusalStabilityProbe, ablate_direction

__all__ = [
    "ContextItem",
    "EvaluationAwarenessProbe",
    "OverwriteDistanceProbe",
    "RefusalStabilityProbe",
    "SycophancyItem",
    "ablate_direction",
]
