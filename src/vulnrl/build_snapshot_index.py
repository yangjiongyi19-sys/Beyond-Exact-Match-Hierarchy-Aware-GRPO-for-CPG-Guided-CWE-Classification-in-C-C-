"""Map dataset functions to root METHOD nodes in reusable project CPG snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .cpg_harness import CpgSnapshot
from .prepare_data import _read_records


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _project_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def build_index(
    dataset_paths: str | Path | list[str | Path],
    project_map_path: str | Path,
    snapshot_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    snapshot_root = Path(snapshot_dir).resolve()
    project_map = json.loads(Path(project_map_path).read_text(encoding="utf-8"))
    graph_by_project = {_project_key(project): str(path) for project, path in project_map.items()}
    cache: dict[str, CpgSnapshot] = {}
    samples: dict[str, dict[str, Any]] = {}
    unmatched: list[dict[str, str]] = []
    input_paths = dataset_paths if isinstance(dataset_paths, list) else [dataset_paths]
    records = [row for dataset_path in input_paths for row in _read_records(dataset_path)]

    for row in records:
        sample_id = str(row.get("sample_id", ""))
        project = _project_key(row.get("project"))
        if not sample_id or not project or project not in graph_by_project:
            unmatched.append({"sample_id": sample_id, "reason": "project_missing_from_map"})
            continue
        relative_graph_path = Path(graph_by_project[project])
        graph_path = (snapshot_root / relative_graph_path).resolve()
        if snapshot_root not in graph_path.parents:
            raise ValueError(f"Project CPG path escapes snapshot directory: {relative_graph_path}")
        if not graph_path.exists():
            unmatched.append({"sample_id": sample_id, "reason": "project_cpg_file_missing"})
            continue
        path_key = str(graph_path)
        if path_key not in cache:
            cache[path_key] = CpgSnapshot.load(graph_path)
        snapshot = cache[path_key]
        target_code = _norm(str(row.get("code", "")))
        target_hash = hashlib.sha256(target_code.encode("utf-8")).hexdigest()
        target_file = str(row.get("file_name") or "").replace("\\", "/").lower()
        target_name = str(row.get("function_name") or "").lower()
        scored: list[tuple[int, str]] = []
        for node_id, node in snapshot.nodes.items():
            if node.kind.upper() != "METHOD":
                continue
            code = _norm(node.code)
            score = 0
            if code:
                code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
                if code_hash == target_hash:
                    score += 100
                elif target_code and (target_code in code or code in target_code):
                    score += 60
            if target_file and node.file and node.file.replace("\\", "/").lower().endswith(target_file):
                score += 20
            if target_name and node.function and target_name in node.function.lower():
                score += 10
            if score:
                scored.append((score, node_id))
        if not scored:
            unmatched.append({"sample_id": sample_id, "reason": "no_method_node_match"})
            continue
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score = scored[0][0]
        best_ids = [node_id for score, node_id in scored if score == best_score]
        if len(best_ids) != 1:
            unmatched.append({"sample_id": sample_id, "reason": "ambiguous_method_node_match"})
            continue
        method_id = best_ids[0]
        ast_frontier = [method_id]
        ast_seen = {method_id}
        candidates: list[tuple[int, str]] = []
        priority = {
            "METHOD_PARAMETER_IN": 0, "PARAMETER": 0, "CONTROL_STRUCTURE": 1,
            "CALL": 2, "IDENTIFIER": 3, "LOCAL": 4, "RETURN": 5,
        }
        while ast_frontier:
            current = ast_frontier.pop()
            for child in snapshot.outgoing.get(current, {}).get("AST", set()):
                if child in ast_seen:
                    continue
                ast_seen.add(child)
                ast_frontier.append(child)
                node = snapshot.nodes.get(child)
                if node and node.kind.upper() in priority and node.code.strip():
                    candidates.append((priority[node.kind.upper()], child))
        candidates.sort(key=lambda item: (item[0], item[1]))
        entry_node_ids = [method_id] + [node_id for _, node_id in candidates[:11]]
        samples[sample_id] = {
            "snapshot": str(relative_graph_path).replace("\\", "/"),
            "entry_node_ids": entry_node_ids,
        }

    index = {"schema_version": "cpg-snapshot-index-v1", "samples": samples}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "dataset_files": [str(path) for path in input_paths],
        "dataset_rows": len(records),
        "indexed": len(samples),
        "unmatched": len(unmatched),
        "unmatched_rows": unmatched,
        "index_out": str(out),
        "index_directory_must_equal_snapshot_directory": str(out.parent.resolve()) == str(snapshot_root),
    }
    out.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, action="append", help="Normalized dataset JSONL; repeat for each split")
    parser.add_argument("--project-map", required=True, help="JSON mapping dataset project name to snapshot-relative JSON")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out", required=True, help="Write index.json inside snapshot-dir")
    args = parser.parse_args()
    print(json.dumps(build_index(args.dataset, args.project_map, args.snapshot_dir, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

