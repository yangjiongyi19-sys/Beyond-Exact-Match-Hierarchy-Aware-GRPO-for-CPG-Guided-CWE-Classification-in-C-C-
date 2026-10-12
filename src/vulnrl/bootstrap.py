"""Paired bootstrap confidence interval for method A minus method B."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .evaluate import _jsonl


def paired_bootstrap(
    first_path: str | Path,
    second_path: str | Path,
    value: str = "alpha_penalty",
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    first_rows = {str(row["sample_id"]): row for row in _jsonl(first_path)}
    second_rows = {str(row["sample_id"]): row for row in _jsonl(second_path)}
    shared = sorted(set(first_rows) & set(second_rows))
    paired = []
    for sample_id in shared:
        a, b = first_rows[sample_id].get(value), second_rows[sample_id].get(value)
        if a is None or b is None:
            continue
        paired.append((float(a), float(b)))
    if not paired:
        raise ValueError(f"No paired numeric {value!r} observations found")
    differences = [a - b for a, b in paired]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        draw = [differences[rng.randrange(len(differences))] for _ in differences]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    low_index = max(0, int(0.025 * iterations))
    high_index = min(iterations - 1, int(0.975 * iterations))
    return {
        "metric": value,
        "difference_definition": "method_a - method_b (negative favors A for penalties; positive favors A for scores)",
        "paired_n": len(paired),
        "mean_difference": observed,
        "percentile_95_ci": [samples[low_index], samples[high_index]],
        "iterations": iterations,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-a", required=True, help="Per-sample JSONL from evaluate.py")
    parser.add_argument("--method-b", required=True)
    parser.add_argument("--value", default="alpha_penalty")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = paired_bootstrap(args.method_a, args.method_b, args.value, args.iterations, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

