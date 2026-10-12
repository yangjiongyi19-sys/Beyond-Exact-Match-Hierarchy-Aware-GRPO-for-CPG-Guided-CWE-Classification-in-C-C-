"""Convert Joern GraphML exports into the immutable normalized CPG JSON format."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attrs(element: ET.Element, key_names: dict[str, str]) -> dict[str, str]:
    result = {}
    for child in element:
        if _local(child.tag) != "data" or child.text is None:
            continue
        name = key_names.get(child.attrib.get("key", ""), child.attrib.get("key", ""))
        result[name.strip().lower()] = child.text.strip()
    return result


def _pick(values: dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = values.get(name.lower())
        if value:
            return value
    return default


def _ancestor_methods(node_id: str, reverse_ast: dict[str, set[str]], method_ids: set[str]) -> set[str]:
    if node_id in method_ids:
        return {node_id}
    visited = {node_id}
    queue = deque([node_id])
    found: set[str] = set()
    while queue and not found:
        for _ in range(len(queue)):
            current = queue.popleft()
            for parent in reverse_ast.get(current, set()):
                if parent in method_ids:
                    found.add(parent)
                elif parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
    return found


def convert_graphml(graphml_path: str | Path, output_path: str | Path, project_id: str) -> dict[str, Any]:
    root = ET.parse(graphml_path).getroot()
    key_names = {}
    for element in root.iter():
        if _local(element.tag) == "key":
            key_names[element.attrib.get("id", "")] = element.attrib.get("attr.name", element.attrib.get("id", ""))
    node_values: dict[str, dict[str, str]] = {}
    edge_rows: list[tuple[str, str, str]] = []
    for element in root.iter():
        tag = _local(element.tag)
        if tag == "node":
            node_id = str(element.attrib.get("id", ""))
            attrs = _attrs(element, key_names)
            attrs.setdefault("id", node_id)
            node_values[node_id] = attrs
        elif tag == "edge":
            source, target = element.attrib.get("source"), element.attrib.get("target")
            if not source or not target:
                continue
            attrs = _attrs(element, key_names)
            relation = _pick(attrs, "label", "type", "edge_label", "name", default=str(element.attrib.get("label", "")))
            edge_rows.append((str(source), str(target), relation.upper()))

    method_ids = {
        node_id for node_id, values in node_values.items()
        if _pick(values, "label", "node_label", "type").upper() == "METHOD"
    }
    reverse_ast: dict[str, set[str]] = defaultdict(set)
    for source, target, relation in edge_rows:
        if relation in {"AST", "AST_EDGE"}:
            reverse_ast[target].add(source)

    rows = []
    method_name: dict[str, str] = {}
    for node_id, values in node_values.items():
        kind = _pick(values, "label", "node_label", "type", default="UNKNOWN").upper()
        name = _pick(values, "full_name", "fullName", "name", "method_full_name")
        if node_id in method_ids:
            method_name[node_id] = name
        file_name = _pick(values, "filename", "file_name", "file", "path")
        line = _pick(values, "line_number", "lineNumber", "line")
        try:
            line_num = int(line) if line and line.isdigit() else None
        except ValueError:
            line_num = None
        rows.append({
            "id": node_id,
            "kind": kind,
            "code": _pick(values, "code", "source", "text"),
            "function": name if node_id in method_ids else "",
            "file": file_name,
            "line": line_num,
        })

    edges: set[tuple[str, str, str]] = set()
    for source, target, relation in edge_rows:
        if source not in node_values or target not in node_values:
            continue
        if relation in {"CALL", "CALLS"}:
            callers = _ancestor_methods(source, reverse_ast, method_ids)
            callees = {target} if target in method_ids else _ancestor_methods(target, reverse_ast, method_ids)
            for caller in callers:
                for callee in callees:
                    edges.add((caller, callee, "CALLS"))
        elif relation in {"AST", "AST_EDGE"}:
            edges.add((source, target, "AST"))
        elif relation in {"REACHING_DEF", "DATAFLOW", "DDG", "DFG"}:
            edges.add((source, target, "DATAFLOW"))
        elif relation in {"CDG", "CONTROLS", "CONTROL_DEPENDENCE"}:
            edges.add((source, target, "CONTROLS"))

    payload = {
        "schema_version": "joern-cpg-snapshot-v1",
        "sample_id": project_id,
        "source_graphml": Path(graphml_path).name,
        "nodes": rows,
        "edges": [{"source": source, "target": target, "kind": kind} for source, target, kind in sorted(edges)],
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"project_id": project_id, "nodes": len(rows), "normalized_edges": len(edges), "out": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphml", required=True, help="Joern joern-export GraphML file")
    parser.add_argument("--project-id", required=True, help="Stable repository snapshot identifier")
    parser.add_argument("--out", required=True, help="Normalized project CPG JSON")
    args = parser.parse_args()
    print(json.dumps(convert_graphml(args.graphml, args.out, args.project_id), indent=2))


if __name__ == "__main__":
    main()

