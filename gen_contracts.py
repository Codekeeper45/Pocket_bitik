"""Stable data contracts shared by the Pocket_bitik image pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass(frozen=True)
class ProviderError(Exception):
    kind: str
    message: str = "Provider request failed"
    status_code: Optional[int] = None
    retryable: bool = False
    def __str__(self) -> str: return self.message

@dataclass(frozen=True)
class SceneContract:
    task_type: str = "create"
    prompt: str = ""
    participants: tuple[str, ...] = ()
    participant_count: Optional[int] = None
    required_text: tuple[str, ...] = ()
    aspect_ratio: Optional[str] = None
    reference_roles: dict[str, str] = field(default_factory=dict)
    immutable_requirements: tuple[str, ...] = ()
    def __post_init__(self):
        if self.task_type not in {"create", "edit", "series"}: raise ValueError("invalid task_type")
        if self.participant_count is not None and self.participant_count < 0: raise ValueError("participant_count must be nonnegative")

@dataclass(frozen=True)
class QAResult:
    status: str
    findings: tuple[dict[str, Any], ...] = ()
    overflow: bool = False
    error: Optional[str] = None
    def __post_init__(self):
        if self.status not in {"pass", "warnings", "unavailable", "fail"}: raise ValueError("invalid QA status")

@dataclass
class GenerationResult:
    image: bytes
    actual_provider: Optional[str] = None
    actual_model: Optional[str] = None
    pixel_size: Optional[tuple[int, int]] = None
    source_prompt: str = ""
    actual_prompt: str = ""
    qa: QAResult = field(default_factory=lambda: QAResult("unavailable"))
    remaining_findings: tuple[dict[str, Any], ...] = ()
    rollback_reasons: tuple[str, ...] = ()
    calls: dict[str, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    delivery_status: str = "not_attempted"

__all__ = ["ProviderError", "SceneContract", "QAResult", "GenerationResult"]
