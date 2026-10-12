"""Deterministic terminal rewards for the hierarchy-aware agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .cwe_taxonomy import CweTaxonomy, canonical_cwe


@dataclass(frozen=True)
class GroundTruth:
    is_vulnerable: bool
    cwe_id: str | None = None


@dataclass(frozen=True)
class Prediction:
    verdict: str
    cwe_id: str | None
    evidence_node_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Prediction":
        verdict = str(value.get("verdict", "")).strip().lower()
        if verdict not in {"safe", "vulnerable"}:
            raise ValueError("verdict must be 'safe' or 'vulnerable'")
        return cls(
            verdict=verdict,
            cwe_id=canonical_cwe(value.get("cwe_id")),
            evidence_node_ids=tuple(str(node) for node in value.get("evidence_node_ids", [])),
        )


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    classification: float
    grounding: float
    efficiency: float
    query_count: int
    valid_evidence_count: int
    submitted_evidence_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prediction_task_score(
    prediction: Prediction,
    truth: GroundTruth,
    taxonomy: CweTaxonomy,
    hierarchy_mode: str = "alpha",
) -> float:
    """Score the final label as one task outcome, without a separate binary term."""
    if prediction.verdict == "safe":
        return 1.0 if not truth.is_vulnerable else 0.0
    if not truth.is_vulnerable or not truth.cwe_id or not prediction.cwe_id:
        return 0.0
    return taxonomy.reward_score(prediction.cwe_id, truth.cwe_id, hierarchy_mode)


def score_episode(
    final: Prediction,
    truth: GroundTruth,
    taxonomy: CweTaxonomy,
    observed_node_ids: Iterable[str],
    query_count: int,
    max_queries: int = 6,
    hierarchy_mode: str = "alpha",
) -> RewardBreakdown:
    """Compute one outcome reward without consulting a language-model judge."""
    classification = prediction_task_score(final, truth, taxonomy, hierarchy_mode)
    observed = set(str(node) for node in observed_node_ids)
    evidence = set(final.evidence_node_ids)
    valid_count = len(evidence & observed)
    if final.verdict == "vulnerable":
        grounding = valid_count / len(evidence) if evidence else 0.0
    else:
        # Safe predictions do not need to cite a vulnerability node.
        grounding = 1.0 if not truth.is_vulnerable else 0.0
    efficiency = 1.0 if max_queries <= 0 else 1.0 - min(max(query_count, 0), max_queries) / max_queries

    # Only the final episode outcome is rewarded. Standard GRPO assigns this
    # scalar to the full sampled trajectory, including the tool decisions.
    total = 0.80 * classification + 0.10 * grounding + 0.10 * efficiency
    return RewardBreakdown(
        total=total,
        classification=classification,
        grounding=grounding,
        efficiency=efficiency,
        query_count=query_count,
        valid_evidence_count=valid_count,
        submitted_evidence_count=len(evidence),
    )

