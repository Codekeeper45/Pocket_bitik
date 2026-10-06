"""Offline, evidence-driven evaluation utilities for image pipeline experiments.

No generation is performed here. Input manifests describe already measured runs;
all quality scores are human/adapter supplied observations, not inferred claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


class EvaluationError(ValueError):
    """Invalid or incomplete evaluation evidence."""


@dataclass(frozen=True)
class OCRObservation:
    expected: str
    observed: str
    exact_match: bool
    adapter: str


def inspect_required_text(expected: str, image_path: str, ocr_adapter: Callable[[str], str] | None = None) -> OCRObservation | None:
    """Optionally inspect text via injected OCR; absent adapter means unavailable."""
    if not expected:
        return None
    if ocr_adapter is None:
        return OCRObservation(expected, "", False, "unavailable")
    observed = ocr_adapter(image_path)
    if not isinstance(observed, str):
        raise EvaluationError("OCR adapter must return text")
    return OCRObservation(expected, observed, observed == expected, getattr(ocr_adapter, "__name__", type(ocr_adapter).__name__))


def validate_profile_options(profile: Any, *, use_mask: bool = False) -> None:
    """Refuse mask experiments unless an explicit provider capability allows them."""
    if use_mask and not bool(getattr(profile, "supports_masks", False)):
        raise EvaluationError("mask experiment requested but provider profile does not support masks")


def measured_upscale(source_size: Sequence[int], output_size: Sequence[int], *, method: str) -> dict[str, Any]:
    """Report measured dimensions without claiming synthetic detail improvement."""
    source = _dimensions(source_size, "source_size")
    output = _dimensions(output_size, "output_size")
    if not method or not isinstance(method, str):
        raise EvaluationError("upscale method must be labeled")
    return {"stage": "upscale", "method": method, "source_dimensions": list(source),
            "output_dimensions": list(output), "dimensions_measured": True,
            "detail_improvement_claimed": False}


def _dimensions(value: Sequence[int], field_name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(x) is not int or x <= 0 for x in value):
        raise EvaluationError(f"{field_name} must contain two positive integer dimensions")
    return value[0], value[1]


def validate_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate measured experiment records and return normalized records.

    Required per-run evidence: id, variant, measured quality dimensions, costs,
    timings, calls, and at least one artifact reference. Missing evidence is not
    silently treated as zero or as a pass.
    """
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise EvaluationError("manifest schema_version must be 1")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise EvaluationError("manifest requires a non-empty runs list")
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    for i, run in enumerate(runs):
        prefix = f"runs[{i}]"
        if not isinstance(run, Mapping):
            raise EvaluationError(f"{prefix} must be an object")
        for key in ("id", "variant"):
            if not isinstance(run.get(key), str) or not run[key].strip():
                raise EvaluationError(f"{prefix}.{key} is required")
        if run["id"] in ids:
            raise EvaluationError(f"duplicate run id: {run['id']}")
        ids.add(run["id"])
        quality = run.get("quality")
        if not isinstance(quality, Mapping) or not quality:
            raise EvaluationError(f"{prefix}.quality requires measured dimensions")
        for name, score in quality.items():
            if not isinstance(name, str) or type(score) not in (int, float) or not 0 <= score <= 1:
                raise EvaluationError(f"{prefix}.quality.{name} must be a measured score in [0,1]")
        cost = run.get("cost_usd")
        if type(cost) not in (int, float) or cost < 0:
            raise EvaluationError(f"{prefix}.cost_usd must be measured and non-negative")
        timing = run.get("timing_ms")
        if not isinstance(timing, Mapping) or not timing:
            raise EvaluationError(f"{prefix}.timing_ms requires measured stage timings")
        for stage, ms in timing.items():
            if type(ms) not in (int, float) or ms < 0:
                raise EvaluationError(f"{prefix}.timing_ms.{stage} must be non-negative")
        calls = run.get("calls")
        if not isinstance(calls, Mapping) or not calls or any(type(v) is not int or v < 0 for v in calls.values()):
            raise EvaluationError(f"{prefix}.calls requires non-negative integer call counts")
        artifacts = run.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts or any(not isinstance(x, str) or not x for x in artifacts):
            raise EvaluationError(f"{prefix}.artifacts requires at least one artifact path/reference")
        normalized.append(dict(run))
    return normalized


def summarize(manifest: Mapping[str, Any]) -> dict[str, Any]:
    runs = validate_manifest(manifest)
    variants: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        variants.setdefault(run["variant"], []).append(run)
    output = {}
    for name, group in variants.items():
        dims = sorted(set.intersection(*(set(r["quality"]) for r in group)))
        output[name] = {
            "n": len(group),
            "mean_quality": {d: sum(float(r["quality"][d]) for r in group) / len(group) for d in dims},
            "mean_cost_usd": sum(float(r["cost_usd"]) for r in group) / len(group),
            "mean_timing_ms": {stage: sum(float(r["timing_ms"].get(stage, 0)) for r in group) / len(group)
                                for stage in sorted(set().union(*(r["timing_ms"].keys() for r in group)))},
            "mean_calls": {stage: sum(r["calls"].get(stage, 0) for r in group) / len(group)
                           for stage in sorted(set().union(*(r["calls"].keys() for r in group)))},
        }
    return {"runs": len(runs), "variants": output}


@dataclass(frozen=True)
class ExperimentOptions:
    """Candidate sampling remains disabled unless explicitly opted in."""
    candidate_count: int = 1
    enabled: bool = False

    def __post_init__(self) -> None:
        if type(self.candidate_count) is not int or self.candidate_count < 1:
            raise EvaluationError("candidate_count must be a positive integer")
        if not self.enabled and self.candidate_count != 1:
            raise EvaluationError("candidate_count > 1 requires explicit enabled=True")


def candidate_count_for(options: ExperimentOptions) -> int:
    return options.candidate_count if options.enabled else 1


__all__ = ["EvaluationError", "OCRObservation", "inspect_required_text", "validate_profile_options",
           "measured_upscale", "validate_manifest", "summarize", "ExperimentOptions", "candidate_count_for"]

if __name__ == "__main__":
    import argparse, json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="path to offline evaluation manifest JSON")
    args = parser.parse_args()
    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    print(json.dumps(summarize(data), indent=2, sort_keys=True))
