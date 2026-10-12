"""Normalize PrimeVul and BigVul with conservative cross-split deduplication."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _lookup(row: dict[str, Any], aliases: Iterable[str], override: str | None = None) -> Any:
    if override:
        if override not in row:
            raise ValueError(f"Requested field {override!r} not present; fields: {sorted(row)}")
        return row[override]
    indexed = {_key(str(k)): v for k, v in row.items()}
    for alias in aliases:
        candidate = _key(alias)
        if candidate in indexed:
            return indexed[candidate]
    return None


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".gz":
        opener = gzip.open
        inner_suffix = path.with_suffix("").suffix.lower()
    else:
        opener = open
        inner_suffix = path.suffix.lower()
    if inner_suffix in {".jsonl", ".ndjson"}:
        with opener(path, "rt", encoding="utf-8-sig") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if inner_suffix == ".json":
        with opener(path, "rt", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            for key in ("data", "records", "items"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list):
            raise ValueError(f"Expected a list of records in {path}")
        return value
    if inner_suffix == ".csv":
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Supported dataset formats are .jsonl, .json, .csv and .gz variants: {path}")


def _bool_label(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "vulnerable", "vul", "positive"}:
        return True
    if normalized in {"0", "false", "no", "safe", "non-vulnerable", "non_vulnerable", "negative"}:
        return False
    raise ValueError(f"Cannot interpret binary vulnerability label: {value!r}")


def _normalize_split(value: Any) -> str:
    split = str(value or "").strip().lower()
    aliases = {"valid": "val", "validation": "val", "dev": "val", "training": "train", "testing": "test"}
    split = aliases.get(split, split)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unrecognized PrimeVul split {value!r}; expected train/val/test")
    return split


def _normalize_code(code: str) -> str:
    # Preserve comments and literals; only collapse whitespace to avoid merging
    # functions whose security-relevant tokens differ.
    return re.sub(r"\s+", " ", code).strip()


def _cwe_id(value: Any) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    matches = re.findall(r"CWE\s*[-:]?\s*(\d+)", str(value), flags=re.IGNORECASE)
    if not matches and str(value).strip().isdigit():
        matches = [str(value).strip()]
    normalized = sorted({f"CWE-{int(item)}" for item in matches})
    return (normalized[0] if normalized else None), len(normalized) > 1


def _apply_bigvul_audit(rows: list[dict[str, Any]], audit_path: str | Path | None) -> tuple[int, list[str]]:
    if not audit_path:
        return 0, []
    audited_records = _read_records(audit_path)
    index: dict[str, str] = {}
    for audit_row in audited_records:
        sample_id = str(_lookup(audit_row, ("sample_id", "id")) or "")
        cwe_value = _lookup(audit_row, ("audited_cwe_id", "cwe_id", "cwe", "CWE ID"))
        cwe_id, multi = _cwe_id(cwe_value)
        if not sample_id or not cwe_id or multi:
            continue
        index[sample_id] = cwe_id
    matched = 0
    for row in rows:
        if row["sample_id"] in index:
            row["cwe_id"] = index[row["sample_id"]]
            row["cwe_raw"] = index[row["sample_id"]]
            row["cwe_multilabel"] = False
            row["cwe_label_status"] = "audited"
            matched += 1
    unmatched = sorted(set(index) - {row["sample_id"] for row in rows})
    return matched, unmatched


def _standardize(
    row: dict[str, Any],
    source: str,
    ordinal: int,
    split: str | None = None,
    code_field: str | None = None,
    label_field: str | None = None,
    cwe_field: str | None = None,
    pair_field: str | None = None,
) -> dict[str, Any]:
    code = _lookup(
        row,
        ("functionSource", "func_before", "function", "func", "code", "source_code"),
        code_field,
    )
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"{source} row {ordinal}: missing source code")
    label_value = _lookup(row, ("target", "vul", "is_vulnerable", "vulnerable", "label"), label_field)
    if label_value is None:
        raise ValueError(f"{source} row {ordinal}: missing binary vulnerability label")
    cwe_raw = _lookup(row, ("cwe", "CWE ID", "CWE-ID", "weakness_id"), cwe_field)
    cwe_id, cwe_multilabel = _cwe_id(cwe_raw)
    project = _lookup(row, ("project", "repo", "repository", "repo_name"))
    cve_id = _lookup(row, ("cve_id", "cve", "CVE ID"))
    commit = _lookup(row, ("commit_id", "commit", "hash", "fix_commit"))
    file_name = _lookup(row, ("file_name", "filename", "file", "path"))
    pair_id = _lookup(
        row,
        ("pair_id", "pair", "sample_pair_id", "pair_idx", "pair_index"),
        pair_field,
    )
    function_name = _lookup(row, ("func_name", "function_name", "name"))
    normalized = _normalize_code(code)
    code_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return {
        "sample_id": f"{source}-{ordinal}-{code_hash[:12]}",
        "source": source,
        "split": split or "train",
        "code": code,
        "code_sha256": code_hash,
        "is_vulnerable": _bool_label(label_value),
        "cwe_id": cwe_id,
        "cwe_raw": str(cwe_raw) if cwe_raw is not None else None,
        "cwe_multilabel": cwe_multilabel,
        "cwe_label_status": "unreviewed" if source == "bigvul" and cwe_id else "source_label",
        "project": str(project).strip() if project else None,
        "cve_id": str(cve_id).strip().upper() if cve_id else None,
        "commit_id": str(commit).strip().lower() if commit else None,
        "file_name": str(file_name).replace("\\", "/").strip() if file_name else None,
        "pair_id": str(pair_id).strip() if pair_id is not None and str(pair_id).strip() else None,
        "function_name": str(function_name).strip() if function_name else None,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _priority(row: dict[str, Any]) -> tuple[int, int]:
    split_priority = {"test": 0, "val": 1, "train": 2}.get(row["split"], 3)
    source_priority = 0 if row["source"] == "primevul" else 1
    return split_priority, source_priority


def _leakage_keys(row: dict[str, Any]) -> set[str]:
    """Comparable provenance keys for conservative cross-split exclusions."""
    keys = {f"code:{row['code_sha256']}"}
    if row.get("cve_id"):
        keys.add(f"cve:{_key(str(row['cve_id']))}")
    if row.get("commit_id"):
        keys.add(f"commit:{_key(str(row['commit_id']))}")
    if row.get("project") and row.get("file_name") and row.get("function_name"):
        location = "/".join((
            _key(str(row["project"])),
            _key(str(row["file_name"]).replace("\\", "/")),
            _key(str(row["function_name"])),
        ))
        keys.add(f"location:{location}")
    return keys


def prepare(
    primevul_path: str | Path,
    bigvul_path: str | Path,
    out_dir: str | Path,
    *,
    augment_bigvul: bool = True,
    prime_code_field: str | None = None,
    big_code_field: str | None = None,
    prime_label_field: str | None = None,
    big_label_field: str | None = None,
    prime_cwe_field: str | None = None,
    big_cwe_field: str | None = None,
    prime_split_field: str | None = None,
    prime_pair_field: str | None = None,
    bigvul_cwe_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    prime_rows = _read_records(primevul_path)
    big_rows = _read_records(bigvul_path)
    prime: list[dict[str, Any]] = []
    for ordinal, row in enumerate(prime_rows):
        split_value = _lookup(row, ("split", "partition", "subset"), prime_split_field)
        if split_value is None:
            raise ValueError("PrimeVul split field was not found; pass --prime-split-field explicitly")
        prime.append(_standardize(
            row, "primevul", ordinal, _normalize_split(split_value), prime_code_field,
            prime_label_field, prime_cwe_field, prime_pair_field,
        ))
    bigvul = [
        _standardize(row, "bigvul", ordinal, "train", big_code_field, big_label_field, big_cwe_field)
        for ordinal, row in enumerate(big_rows)
    ]
    audited_bigvul_count, unmatched_audit_ids = _apply_bigvul_audit(bigvul, bigvul_cwe_audit_path)

    all_rows = prime + bigvul
    best_priority_by_key: dict[str, tuple[int, int]] = {}
    for row in all_rows:
        priority = _priority(row)
        for key in _leakage_keys(row):
            if key not in best_priority_by_key or priority < best_priority_by_key[key]:
                best_priority_by_key[key] = priority

    eligible_rows = []
    leakage_exclusions: dict[str, int] = defaultdict(int)
    leakage_excluded_rows: list[dict[str, Any]] = []
    for row in all_rows:
        priority = _priority(row)
        overlapping = sorted(
            key for key in _leakage_keys(row)
            if best_priority_by_key[key] < priority
        )
        if overlapping:
            leakage_exclusions[f"{row['source']}:{row['split']}"] += 1
            leakage_excluded_rows.append({
                "sample_id": row["sample_id"],
                "source": row["source"],
                "split": row["split"],
                "overlapping_provenance_keys": overlapping,
            })
        else:
            eligible_rows.append(row)

    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        by_hash[row["code_sha256"]].append(row)
    conflicts = []
    dropped_duplicates: dict[str, int] = defaultdict(int)
    winners: dict[str, dict[str, Any]] = {}
    for code_hash, group in by_hash.items():
        ordered = sorted(group, key=_priority)
        winner = ordered[0]
        prime_group = [row for row in group if row["source"] == "primevul"]
        prime_labels = {(row["is_vulnerable"], row["cwe_id"]) for row in prime_group}
        if len(prime_labels) > 1:
            conflicts.append({
                "code_sha256": code_hash,
                "sample_ids": [row["sample_id"] for row in prime_group],
                "labels": sorted([{"is_vulnerable": b, "cwe_id": c} for b, c in prime_labels], key=str),
                "resolution": f"retained {winner['sample_id']} by test > val > train precedence",
            })
        winners[code_hash] = winner
        for row in group:
            if row is not winner:
                dropped_duplicates[f"{row['source']}:{row['split']}"] += 1

    prime_clean = [row for row in winners.values() if row["source"] == "primevul"]
    prime_train = sorted((r for r in prime_clean if r["split"] == "train"), key=lambda r: r["sample_id"])
    prime_val = sorted((r for r in prime_clean if r["split"] == "val"), key=lambda r: r["sample_id"])
    prime_test = sorted((r for r in prime_clean if r["split"] == "test"), key=lambda r: r["sample_id"])
    bigvul_novel = [row for row in winners.values() if row["source"] == "bigvul"]
    bigvul_eval = sorted(bigvul_novel, key=lambda r: r["sample_id"])
    augmented_train = prime_train + (bigvul_novel if augment_bigvul else [])
    augmented_train.sort(key=lambda r: (r["source"], r["sample_id"]))

    prime_train_projects = {r["project"].lower() for r in prime_train if r["project"]}
    all_train_projects = {r["project"].lower() for r in augmented_train if r["project"]}
    cross_project = [r for r in prime_test if r["project"] and r["project"].lower() not in all_train_projects]
    prime_only_cross_project = [r for r in prime_test if r["project"] and r["project"].lower() not in prime_train_projects]

    out = Path(out_dir)
    counts = {
        "primevul_train": _write_jsonl(out / "primevul_train.jsonl", prime_train),
        "primevul_val": _write_jsonl(out / "primevul_val.jsonl", prime_val),
        "primevul_test": _write_jsonl(out / "primevul_test.jsonl", prime_test),
        "primevul_test_cross_project_vs_augmented_train": _write_jsonl(out / "primevul_test_cross_project.jsonl", cross_project),
        "primevul_test_cross_project_vs_primevul_train": _write_jsonl(out / "primevul_test_cross_project_primevul_only.jsonl", prime_only_cross_project),
        "bigvul_eval_deduplicated_against_primevul": _write_jsonl(out / "bigvul_eval.jsonl", bigvul_eval),
    }
    if augment_bigvul:
        counts["train_augmented"] = _write_jsonl(out / "train_augmented.jsonl", augmented_train)
    manifest = {
        "schema_version": "vulnrl-dataset-v1",
        "primevul_input": str(primevul_path),
        "bigvul_input": str(bigvul_path),
        "bigvul_cwe_labels_are_unreviewed": True,
        "bigvul_audit_file": str(bigvul_cwe_audit_path) if bigvul_cwe_audit_path else None,
        "bigvul_cwe_labels_audited_count": audited_bigvul_count,
        "unmatched_bigvul_audit_sample_ids": unmatched_audit_ids,
        "bigvul_augmentation_enabled": augment_bigvul,
        "dedup_key": "normalized function SHA-256 plus exact CVE, commit, and project/file/function provenance keys when present",
        "dedup_precedence": "PrimeVul test > PrimeVul val > PrimeVul train > BigVul train",
        "counts": counts,
        "duplicates_dropped_by_loser_split": dict(dropped_duplicates),
        "cross_split_provenance_exclusions": dict(leakage_exclusions),
        "cross_split_provenance_excluded_rows": leakage_excluded_rows,
        "primevul_label_conflicts": conflicts,
        "primevul_test_duplicate_code_hashes_in_any_train_source": len({r["code_sha256"] for r in prime_test} & {r["code_sha256"] for r in (augmented_train if augment_bigvul else prime_train)}),
        "primevul_cross_project_rows_relative_to_augmented_train": len(cross_project),
        "primevul_cross_project_rows_relative_to_primevul_train": len(prime_only_cross_project),
        "warning": "BigVul CWE labels remain unreviewed. Do not use them as confirmed CWE reward labels without a separate audit.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primevul", required=True)
    parser.add_argument("--bigvul", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-bigvul-augmentation", action="store_true")
    parser.add_argument("--prime-code-field")
    parser.add_argument("--big-code-field")
    parser.add_argument("--prime-label-field")
    parser.add_argument("--big-label-field")
    parser.add_argument("--prime-cwe-field")
    parser.add_argument("--big-cwe-field")
    parser.add_argument("--prime-split-field")
    parser.add_argument("--prime-pair-field", help="Source field that identifies vulnerable/fixed PrimeVul pairs")
    parser.add_argument("--bigvul-cwe-audit", help="Optional JSONL/CSV sample_id to reviewed cwe_id mapping")
    args = parser.parse_args()
    manifest = prepare(
        args.primevul,
        args.bigvul,
        args.out_dir,
        augment_bigvul=not args.no_bigvul_augmentation,
        prime_code_field=args.prime_code_field,
        big_code_field=args.big_code_field,
        prime_label_field=args.prime_label_field,
        big_label_field=args.big_label_field,
        prime_cwe_field=args.prime_cwe_field,
        big_cwe_field=args.big_cwe_field,
        prime_split_field=args.prime_split_field,
        prime_pair_field=args.prime_pair_field,
        bigvul_cwe_audit_path=args.bigvul_cwe_audit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

