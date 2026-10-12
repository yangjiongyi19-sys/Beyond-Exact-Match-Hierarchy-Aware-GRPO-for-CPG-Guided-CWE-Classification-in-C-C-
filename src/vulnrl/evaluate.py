"""Metrics for CWE agent predictions and PrimeVul held-out evaluations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .cwe_taxonomy import CweTaxonomy, canonical_cwe
from .prepare_data import _read_records


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        scores.append(_prf(tp, fp, fn)["f1"])
    return sum(scores) / len(scores)


def evaluate(
    labels_path: str | Path,
    predictions_path: str | Path,
    taxonomy: CweTaxonomy,
    train_labels_path: str | Path | None = None,
    require_audited_cwe: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = {str(row["sample_id"]): row for row in _read_records(labels_path)}
    predictions = {}
    for row in _jsonl(predictions_path):
        if "final" in row and isinstance(row["final"], dict):
            row = {**row, **row["final"]}
        predictions[str(row["sample_id"])] = row
    sample_ids = sorted(set(labels) & set(predictions))
    if not sample_ids:
        raise ValueError("No matching sample_id values between labels and predictions")

    tp = fp = fn = tn = 0
    true_cwes: list[str] = []
    pred_cwes: list[str] = []
    alpha_penalties: list[float] = []
    alpha_scores: list[float] = []
    unseen_penalties: list[float] = []
    unseen_scores: list[float] = []
    seen_penalties: list[float] = []
    seen_scores: list[float] = []
    training_cwes: set[str] = set()
    if train_labels_path:
        training_cwes = {
            cwe for row in _read_records(train_labels_path)
            if (cwe := canonical_cwe(row.get("cwe_id"))) is not None
        }

    query_counts = []
    valid_evidence = 0
    submitted_evidence = 0
    paired: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    per_sample = []
    missing_cwe_truth = 0
    invalid_prediction_count = 0
    for sample_id in sample_ids:
        truth = labels[sample_id]
        prediction = predictions[sample_id]
        target_vuln = bool(truth.get("is_vulnerable"))
        verdict = str(prediction.get("verdict", "")).lower()
        pred_vuln = verdict == "vulnerable"
        if verdict not in {"safe", "vulnerable"}:
            invalid_prediction_count += 1
        if target_vuln and pred_vuln:
            tp += 1
        elif target_vuln and not pred_vuln:
            fn += 1
        elif not target_vuln and pred_vuln:
            fp += 1
        else:
            tn += 1
        sample_correct = target_vuln == pred_vuln
        if truth.get("pair_id"):
            paired[str(truth["pair_id"])].append((sample_correct, target_vuln))

        true_cwe = canonical_cwe(truth.get("cwe_id"))
        if require_audited_cwe and truth.get("cwe_label_status") != "audited":
            true_cwe = None
        predicted_cwe = canonical_cwe(prediction.get("cwe_id")) if pred_vuln else None
        if target_vuln and true_cwe:
            true_cwes.append(true_cwe)
            pred_cwes.append(predicted_cwe or "<MISSING>")
            penalty = taxonomy.penalty(predicted_cwe, true_cwe)
            score = taxonomy.score(predicted_cwe, true_cwe)
            alpha_penalties.append(penalty)
            alpha_scores.append(score)
            if training_cwes:
                if true_cwe in training_cwes:
                    seen_penalties.append(penalty)
                    seen_scores.append(score)
                else:
                    unseen_penalties.append(penalty)
                    unseen_scores.append(score)
        elif target_vuln:
            missing_cwe_truth += 1

        evidence = prediction.get("evidence_node_ids", []) or []
        observed = set(str(node) for node in prediction.get("observed_node_ids", []) or [])
        if not observed and isinstance(prediction.get("observations"), list):
            observed = {
                str(node["node_id"])
                for obs in prediction["observations"] if isinstance(obs, dict)
                for node in obs.get("nodes", []) if isinstance(node, dict) and node.get("node_id") is not None
            }
        evidence = [str(node) for node in evidence]
        submitted_evidence += len(evidence)
        valid_evidence += sum(node in observed for node in evidence)
        query_count = int(prediction.get("query_count", prediction.get("tool_call_count", 0)) or 0)
        query_counts.append(query_count)
        per_sample.append({
            "sample_id": sample_id,
            "binary_correct": sample_correct,
            "exact_cwe_correct": bool(target_vuln and true_cwe and predicted_cwe == true_cwe),
            "alpha_penalty": alpha_penalties[-1] if target_vuln and true_cwe else None,
            "alpha_score": alpha_scores[-1] if target_vuln and true_cwe else None,
            "query_count": query_count,
        })

    complete_pairs = [
        all(correct for correct, _ in rows)
        for rows in paired.values()
        if len(rows) >= 2 and {label for _, label in rows} == {False, True}
    ]
    report: dict[str, Any] = {
        "n_evaluated": len(sample_ids),
        "n_label_rows": len(labels),
        "n_prediction_rows": len(predictions),
        "binary": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, **_prf(tp, fp, fn)},
        "cwe": {
            "n_vulnerable_with_cwe": len(true_cwes),
            "exact_macro_f1": _macro_f1(true_cwes, pred_cwes),
            "exact_accuracy": sum(a == b for a, b in zip(true_cwes, pred_cwes)) / len(true_cwes) if true_cwes else 0.0,
            "mean_alpha_penalty_lower_is_better": sum(alpha_penalties) / len(alpha_penalties) if alpha_penalties else None,
            "mean_normalized_alpha_score_higher_is_better": sum(alpha_scores) / len(alpha_scores) if alpha_scores else None,
            "missing_cwe_truth_rows_excluded": missing_cwe_truth,
        },
        "cwe_seen_unseen": {
            "training_cwe_count": len(training_cwes),
            "seen": {"n": len(seen_penalties), "mean_alpha_penalty": sum(seen_penalties) / len(seen_penalties) if seen_penalties else None,
                     "mean_normalized_score": sum(seen_scores) / len(seen_scores) if seen_scores else None},
            "unseen": {"n": len(unseen_penalties), "mean_alpha_penalty": sum(unseen_penalties) / len(unseen_penalties) if unseen_penalties else None,
                       "mean_normalized_score": sum(unseen_scores) / len(unseen_scores) if unseen_scores else None},
        },
        "primevul_pairing": {
            "pair_groups": len(paired),
            "complete_vulnerable_fixed_pairs": len(complete_pairs),
            "both_sides_correct_rate": sum(complete_pairs) / len(complete_pairs) if complete_pairs else None,
        },
        "agent": {
            "mean_tool_queries": sum(query_counts) / len(query_counts) if query_counts else 0.0,
            "valid_evidence_node_rate": valid_evidence / submitted_evidence if submitted_evidence else None,
            "valid_evidence_nodes": valid_evidence,
            "submitted_evidence_nodes": submitted_evidence,
            "invalid_prediction_rows": invalid_prediction_count,
        },
        "taxonomy": {"source_name": taxonomy.source_name, "source_sha256": taxonomy.source_sha256,
                     "node_count": len(taxonomy.names), "max_penalty": taxonomy.max_penalty()},
        "cwe_audit_filter": "audited_only" if require_audited_cwe else "all_available_labels",
    }
    return report, per_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="Normalized evaluation JSONL")
    parser.add_argument("--predictions", required=True, help="Agent prediction JSONL")
    parser.add_argument("--taxonomy", required=True, help="Frozen CWE taxonomy JSON")
    parser.add_argument("--train-labels", help="Training JSONL for seen/unseen CWE breakdown")
    parser.add_argument("--require-audited-cwe", action="store_true", help="Exclude unreviewed CWE labels from CWE metrics")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-sample-out")
    args = parser.parse_args()
    taxonomy = CweTaxonomy.from_json(args.taxonomy)
    report, per_sample = evaluate(
        args.labels,
        args.predictions,
        taxonomy,
        args.train_labels,
        require_audited_cwe=args.require_audited_cwe,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.per_sample_out:
        sample_out = Path(args.per_sample_out)
        sample_out.parent.mkdir(parents=True, exist_ok=True)
        sample_out.write_text("".join(json.dumps(row) + "\n" for row in per_sample), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

