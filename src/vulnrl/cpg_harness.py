"""Read-only, bounded CPG tools over precomputed Joern snapshots."""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SNAPSHOT_CACHE: OrderedDict[tuple[str, int, int], "CpgSnapshot"] = OrderedDict()
_SNAPSHOT_CACHE_LOCK = threading.RLock()


@dataclass(frozen=True)
class CpgNode:
    node_id: str
    code: str
    kind: str = "UNKNOWN"
    function: str = ""
    file: str = ""
    line: int | None = None


class CpgSnapshot:
    """Immutable view of one precomputed project CPG and a function's entry nodes."""

    def __init__(self, payload: dict[str, Any], sample_id: str | None = None, entry_node_ids: list[str] | None = None):
        self.sample_id = str(sample_id if sample_id is not None else payload["sample_id"])
        self.nodes: dict[str, CpgNode] = {}
        for item in payload.get("nodes", []):
            node_id = str(item["id"])
            self.nodes[node_id] = CpgNode(
                node_id=node_id,
                code=str(item.get("code", "")),
                kind=str(item.get("kind", "UNKNOWN")),
                function=str(item.get("function", "")),
                file=str(item.get("file", "")),
                line=int(item["line"]) if item.get("line") is not None else None,
            )
        roots = entry_node_ids if entry_node_ids is not None else payload.get("entry_node_ids", [])
        self.entry_node_ids = [str(node_id) for node_id in roots]
        self.outgoing: dict[str, dict[str, set[str]]] = {}
        self.incoming: dict[str, dict[str, set[str]]] = {}
        for edge in payload.get("edges", []):
            source, target = str(edge["source"]), str(edge["target"])
            relation = str(edge["kind"]).upper()
            if source not in self.nodes or target not in self.nodes:
                continue
            self.outgoing.setdefault(source, {}).setdefault(relation, set()).add(target)
            self.incoming.setdefault(target, {}).setdefault(relation, set()).add(source)

    @classmethod
    def load(
        cls,
        path: str | Path,
        sample_id: str | None = None,
        entry_node_ids: list[str] | None = None,
    ) -> "CpgSnapshot":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), sample_id, entry_node_ids)

    def view_for(self, sample_id: str, entry_node_ids: list[str] | None = None) -> "CpgSnapshot":
        """Share immutable indexes while using the selected function's entry nodes."""
        view = object.__new__(CpgSnapshot)
        view.sample_id = sample_id
        view.nodes = self.nodes
        view.entry_node_ids = list(self.entry_node_ids if entry_node_ids is None else entry_node_ids)
        view.outgoing = self.outgoing
        view.incoming = self.incoming
        return view

    def _node_rows(self, ids: set[str], limit: int) -> list[dict[str, Any]]:
        rows = []
        for node_id in sorted(ids)[:limit]:
            node = self.nodes.get(node_id)
            if node is None:
                continue
            rows.append({
                "node_id": node.node_id,
                "kind": node.kind,
                "code": node.code[:1200],
                "function": node.function,
                "file": node.file,
                "line": node.line,
            })
        return rows

    def call_neighbors(self, node_id: str, direction: str, limit: int) -> list[dict[str, Any]]:
        if node_id not in self.nodes:
            return []
        if direction == "callees":
            ids = self.outgoing.get(node_id, {}).get("CALLS", set())
        else:
            ids = self.incoming.get(node_id, {}).get("CALLS", set())
        return self._node_rows(ids, limit)

    def initial_nodes(self, limit: int) -> list[dict[str, Any]]:
        ids = set(self.entry_node_ids)
        if not ids:
            ids = {
                node_id for node_id, node in self.nodes.items()
                if node.kind.upper() in {"METHOD", "PARAMETER", "IDENTIFIER"}
            }
        return self._node_rows(ids, limit)

    def dataflow_slice(self, node_id: str, direction: str, depth: int, limit: int) -> list[dict[str, Any]]:
        if node_id not in self.nodes:
            return []
        frontier = {node_id}
        visited = {node_id}
        for _ in range(max(1, min(depth, 5))):
            following: set[str] = set()
            for current in frontier:
                if direction in {"forward", "both"}:
                    following.update(self.outgoing.get(current, {}).get("DATAFLOW", set()))
                if direction in {"backward", "both"}:
                    following.update(self.incoming.get(current, {}).get("DATAFLOW", set()))
            following -= visited
            visited.update(following)
            frontier = following
            if not frontier or len(visited) >= limit + 1:
                break
        visited.discard(node_id)
        return self._node_rows(visited, limit)


def _load_cached_snapshot(path: Path) -> CpgSnapshot:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cache_size = max(1, int(os.environ.get("VULNRL_CPG_SNAPSHOT_CACHE", "2")))
    with _SNAPSHOT_CACHE_LOCK:
        cached = _SNAPSHOT_CACHE.get(key)
        if cached is not None:
            _SNAPSHOT_CACHE.move_to_end(key)
            return cached
        loaded = CpgSnapshot.load(path)
        _SNAPSHOT_CACHE[key] = loaded
        while len(_SNAPSHOT_CACHE) > cache_size:
            _SNAPSHOT_CACHE.popitem(last=False)
        return loaded


class CpgHarness:
    """Small read-only tool environment with query limits and evidence tracking."""

    TOOL_NAMES = {"get_callers", "get_callees", "dataflow_slice"}

    def __init__(self, snapshot_dir: str | Path, max_queries: int = 3, max_results: int = 12):
        self.snapshot_dir = Path(snapshot_dir)
        self.max_queries = max_queries
        self.max_results = max_results
        self.query_count = 0
        self.sample_id = ""
        self.snapshot: CpgSnapshot | None = None
        self.observed_node_ids: set[str] = set()

    def reset(self, sample_id: str) -> dict[str, Any]:
        index_path = self.snapshot_dir / "index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            entry = index.get("samples", {}).get(sample_id)
            if not isinstance(entry, dict) or not entry.get("snapshot"):
                raise FileNotFoundError(f"No CPG index entry for sample {sample_id}")
            root = self.snapshot_dir.resolve()
            path = (self.snapshot_dir / str(entry["snapshot"])).resolve()
            if root not in path.parents:
                raise ValueError("CPG snapshot index path escapes snapshot directory")
            base_snapshot = _load_cached_snapshot(path)
            self.snapshot = base_snapshot.view_for(sample_id, entry.get("entry_node_ids", []))
        else:
            path = self.snapshot_dir / f"{sample_id}.json"
            base_snapshot = _load_cached_snapshot(path)
            if base_snapshot.sample_id != sample_id:
                raise ValueError(f"Snapshot id mismatch: expected {sample_id}, got {base_snapshot.sample_id}")
            self.snapshot = base_snapshot.view_for(sample_id)
        self.sample_id = sample_id
        self.query_count = 0
        self.observed_node_ids.clear()
        observation = {
            "tool": "initial_cpg",
            "query_count": 0,
            "nodes": self.snapshot.initial_nodes(self.max_results),
        }
        return self._record(observation)

    def _record(self, observation: dict[str, Any]) -> dict[str, Any]:
        for row in observation.get("nodes", []):
            if row.get("node_id") is not None:
                self.observed_node_ids.add(str(row["node_id"]))
        return observation

    def step(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.snapshot is None:
            raise RuntimeError("Call reset(sample_id) before using tools")
        if self.query_count >= self.max_queries:
            return {"error": "query_budget_exhausted", "query_count": self.query_count}
        if tool_name not in self.TOOL_NAMES:
            self.query_count += 1
            return {"error": "unknown_tool", "allowed_tools": sorted(self.TOOL_NAMES)}

        self.query_count += 1
        node_id = str(arguments.get("node_id", ""))
        if tool_name == "get_callers":
            rows = self.snapshot.call_neighbors(node_id, "callers", self.max_results)
        elif tool_name == "get_callees":
            rows = self.snapshot.call_neighbors(node_id, "callees", self.max_results)
        else:
            direction = str(arguments.get("direction", "both")).lower()
            if direction not in {"forward", "backward", "both"}:
                return {"error": "direction must be forward, backward, or both", "query_count": self.query_count}
            rows = self.snapshot.dataflow_slice(
                node_id,
                direction,
                int(arguments.get("depth", 2)),
                self.max_results,
            )
        return self._record({"tool": tool_name, "query_count": self.query_count, "nodes": rows})

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": "get_callers", "description": "List caller methods for a method node.", "parameters": {"node_id": "string"}},
            {"name": "get_callees", "description": "List callee methods for a method node.", "parameters": {"node_id": "string"}},
            {"name": "dataflow_slice", "description": "Read a bounded data-flow slice around a node.", "parameters": {"node_id": "string", "direction": "forward|backward|both", "depth": "integer 1..5"}},
        ]
