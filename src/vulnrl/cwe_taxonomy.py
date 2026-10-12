"""MITRE CWE XML loader and the direction-aware ALPHA penalty."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CWE_RE = re.compile(r"(?:CWE-)?(\d+)", re.IGNORECASE)
_ALLOWED_ABSTRACTIONS = {"Pillar", "Class", "Base", "Variant", "Compound", "Chain"}


def canonical_cwe(value: Any) -> str | None:
    if value is None:
        return None
    match = _CWE_RE.fullmatch(str(value).strip())
    return f"CWE-{int(match.group(1))}" if match else None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


@dataclass
class CweTaxonomy:
    names: dict[str, str]
    abstractions: dict[str, str]
    descriptions: dict[str, str]
    children: dict[str, set[str]]
    source_sha256: str = ""
    source_name: str = ""
    alpha_up: float = 2.0
    alpha_lateral: float = 1.8
    alpha_min_down: float = 1.1
    alpha_oog: float = 2.5
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_mitre_xml(cls, path: str | Path) -> "CweTaxonomy":
        path = Path(path)
        raw = path.read_bytes()
        root = ET.fromstring(raw)
        names: dict[str, str] = {}
        abstractions: dict[str, str] = {}
        descriptions: dict[str, str] = {}
        children: dict[str, set[str]] = {}
        for element in root.iter():
            if _local(element.tag) != "Weakness":
                continue
            cwe = canonical_cwe(element.attrib.get("ID"))
            abstraction = element.attrib.get("Abstraction", "")
            if cwe is None or abstraction not in _ALLOWED_ABSTRACTIONS:
                continue
            names[cwe] = element.attrib.get("Name", "")
            abstractions[cwe] = abstraction
            description_parts = []
            for child in element:
                if _local(child.tag) == "Description" and child.text:
                    description_parts.append(" ".join(child.text.split()))
            descriptions[cwe] = " ".join(description_parts)
            children.setdefault(cwe, set())

        # CWE XML expresses ChildOf from a weakness to its parent. Store edges
        # parent -> child so reachability directly answers ancestor queries.
        for element in root.iter():
            if _local(element.tag) != "Weakness":
                continue
            child_id = canonical_cwe(element.attrib.get("ID"))
            if child_id not in names:
                continue
            for relation in element.iter():
                if _local(relation.tag) != "Related_Weakness":
                    continue
                if relation.attrib.get("Nature", "").lower() != "childof":
                    continue
                parent_id = canonical_cwe(relation.attrib.get("CWE_ID"))
                if parent_id in names and parent_id != child_id:
                    children.setdefault(parent_id, set()).add(child_id)

        if not names:
            raise ValueError(f"No supported CWE Weakness nodes found in {path}")
        return cls(
            names=names,
            abstractions=abstractions,
            descriptions=descriptions,
            children=children,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            source_name=path.name,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CweTaxonomy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            names={k: str(v) for k, v in payload["names"].items()},
            abstractions={k: str(v) for k, v in payload["abstractions"].items()},
            descriptions={k: str(v) for k, v in payload.get("descriptions", {}).items()},
            children={k: set(v) for k, v in payload["children"].items()},
            source_sha256=payload.get("source_sha256", ""),
            source_name=payload.get("source_name", ""),
            alpha_up=float(payload.get("alpha_up", 2.0)),
            alpha_lateral=float(payload.get("alpha_lateral", 1.8)),
            alpha_min_down=float(payload.get("alpha_min_down", 1.1)),
            alpha_oog=float(payload.get("alpha_oog", 2.5)),
        )

    def to_json(self, path: str | Path) -> None:
        payload = {
            "version": "cwe-taxonomy-v1",
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "alpha_up": self.alpha_up,
            "alpha_lateral": self.alpha_lateral,
            "alpha_min_down": self.alpha_min_down,
            "alpha_oog": self.alpha_oog,
            "names": self.names,
            "abstractions": self.abstractions,
            "descriptions": self.descriptions,
            "children": {key: sorted(value) for key, value in self.children.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def parents(self) -> dict[str, set[str]]:
        result = {node: set() for node in self.names}
        for parent, children in self.children.items():
            for child in children:
                result.setdefault(child, set()).add(parent)
        return result

    def _undirected(self) -> dict[str, set[str]]:
        if "undirected" in self._cache:
            return self._cache["undirected"]
        graph = {node: set() for node in self.names}
        for parent, children in self.children.items():
            for child in children:
                graph[parent].add(child)
                graph[child].add(parent)
        self._cache["undirected"] = graph
        return graph

    @staticmethod
    def _bfs(start: str, graph: dict[str, set[str]]) -> dict[str, int]:
        distances = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, ()):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        return distances

    def _descendant_depths(self) -> dict[str, int]:
        if "depths" in self._cache:
            return self._cache["depths"]
        memo: dict[str, int] = {}
        visiting: set[str] = set()

        def depth(node: str) -> int:
            if node in memo:
                return memo[node]
            if node in visiting:
                raise ValueError(f"CWE hierarchy contains a cycle involving {node}")
            visiting.add(node)
            descendants = self.children.get(node, set())
            value = 0 if not descendants else 1 + max(depth(child) for child in descendants)
            visiting.remove(node)
            memo[node] = value
            return value

        for node in self.names:
            depth(node)
        self._cache["depths"] = memo
        return memo

    def _descendants(self) -> dict[str, set[str]]:
        if "descendants" not in self._cache:
            self._cache["descendants"] = {node: set(self._bfs(node, self.children)) - {node} for node in self.names}
        return self._cache["descendants"]

    def _direction(self, predicted: str, truth: str) -> str:
        descendants = self._descendants()
        if truth in descendants.get(predicted, set()):
            return "up"
        if predicted in descendants.get(truth, set()):
            return "down"
        return "lateral"

    def _down_multiplier(self, truth: str, depths: dict[str, int]) -> float:
        abstraction = self.abstractions.get(truth, "")
        if "max_depth_by_abstraction" not in self._cache:
            by_abstraction: dict[str, int] = {}
            for node, level in self.abstractions.items():
                by_abstraction[level] = max(by_abstraction.get(level, 0), depths.get(node, 0))
            self._cache["max_depth_by_abstraction"] = by_abstraction
        max_room = self._cache["max_depth_by_abstraction"].get(abstraction, 0)
        if max_room == 0:
            return self.alpha_lateral
        ratio = depths.get(truth, 0) / max_room
        return self.alpha_lateral - (self.alpha_lateral - self.alpha_min_down) * ratio

    def _graph_stats(self) -> tuple[float, int, float]:
        if "graph_stats" in self._cache:
            return self._cache["graph_stats"]
        graph = self._undirected()
        total_distance = 0
        connected_pairs = 0
        diameter = 0
        nodes = list(self.names)
        for index, node in enumerate(nodes):
            distances = self._bfs(node, graph)
            for other in nodes[index + 1 :]:
                if other in distances:
                    total_distance += distances[other]
                    connected_pairs += 1
                    diameter = max(diameter, distances[other])
        mean = total_distance / connected_pairs if connected_pairs else 1.0
        self._cache["graph_stats"] = (mean, math.ceil(mean), float(diameter))
        return self._cache["graph_stats"]

    def max_penalty(self) -> float:
        if "max_penalty" in self._cache:
            return self._cache["max_penalty"]
        _, out_distance, diameter = self._graph_stats()
        depths = self._descendant_depths()
        graph = self._undirected()
        max_in_graph = 0.0
        for truth in self.names:
            distances = self._bfs(truth, graph)
            for predicted, distance in distances.items():
                if predicted == truth:
                    continue
                direction = self._direction(predicted, truth)
                if direction == "up":
                    multiplier = self.alpha_up
                elif direction == "down":
                    multiplier = self._down_multiplier(truth, depths)
                else:
                    multiplier = self.alpha_lateral
                max_in_graph = max(max_in_graph, distance * multiplier)
        self._cache["max_penalty"] = max(max_in_graph, out_distance * self.alpha_oog, 1.0)
        return self._cache["max_penalty"]

    def penalty(self, predicted: Any, truth: Any) -> float:
        predicted_id = canonical_cwe(predicted)
        truth_id = canonical_cwe(truth)
        if predicted_id is None or truth_id is None:
            return self._graph_stats()[1] * self.alpha_oog
        if predicted_id == truth_id:
            return 0.0
        if predicted_id not in self.names or truth_id not in self.names:
            return self._graph_stats()[1] * self.alpha_oog
        distances = self._bfs(predicted_id, self._undirected())
        if truth_id not in distances:
            return self._graph_stats()[1] * self.alpha_oog
        distance = distances[truth_id]
        direction = self._direction(predicted_id, truth_id)
        if direction == "up":
            multiplier = self.alpha_up
        elif direction == "down":
            multiplier = self._down_multiplier(truth_id, self._descendant_depths())
        else:
            multiplier = self.alpha_lateral
        return distance * multiplier

    def score(self, predicted: Any, truth: Any) -> float:
        """Return 1 - normalized penalty; exact predictions score 1."""
        return max(0.0, min(1.0, 1.0 - self.penalty(predicted, truth) / self.max_penalty()))

    def reward_score(self, predicted: Any, truth: Any, mode: str = "alpha") -> float:
        """Return the main hierarchy score or the exact-match baseline score."""
        predicted_id = canonical_cwe(predicted)
        truth_id = canonical_cwe(truth)
        if mode == "alpha":
            return self.score(predicted_id, truth_id)
        if mode == "exact":
            return float(predicted_id is not None and predicted_id == truth_id)
        raise ValueError(f"Unknown CWE reward mode {mode!r}; expected alpha or exact")

    def lookup(self, query: str, limit: int = 8) -> list[dict[str, str]]:
        terms = {part.lower() for part in re.findall(r"[a-zA-Z0-9_-]+", query) if len(part) > 1}
        ranked: list[tuple[int, str]] = []
        for cwe, name in self.names.items():
            searchable = f"{cwe} {name} {self.descriptions.get(cwe, '')}".lower()
            score = sum(1 for term in terms if term in searchable)
            if score:
                ranked.append((score, cwe))
        ranked.sort(key=lambda item: (-item[0], int(item[1].split("-")[1])))
        return [
            {"cwe_id": cwe, "name": self.names[cwe], "abstraction": self.abstractions[cwe]}
            for _, cwe in ranked[:limit]
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen CWE hierarchy JSON from MITRE's XML export.")
    parser.add_argument("--xml", required=True, help="Path to the MITRE CWE XML download, e.g. CWE 4.20")
    parser.add_argument("--out", required=True, help="Output taxonomy JSON")
    args = parser.parse_args()
    taxonomy = CweTaxonomy.from_mitre_xml(args.xml)
    taxonomy.to_json(args.out)
    print(json.dumps({"nodes": len(taxonomy.names), "sha256": taxonomy.source_sha256, "out": args.out}))


if __name__ == "__main__":
    main()

