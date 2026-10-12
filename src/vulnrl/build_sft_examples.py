"""Build label-supervised, single-turn examples for the ordinary SFT baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .prepare_data import _read_records
from .prompts import SINGLE_TURN_SYSTEM_PROMPT


def build(input_path: str | Path, output_path: str | Path, allow_unreviewed_bigvul: bool = False) -> dict[str, Any]:
    rows = _read_records(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"input": len(rows), "written": 0, "skipped_missing_cwe": 0, "skipped_unreviewed_bigvul": 0,
              "skipped_multilabel": 0, "skipped_nontrain": 0}
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if row.get("split") != "train":
                counts["skipped_nontrain"] += 1
                continue
            if row.get("source") == "bigvul" and row.get("cwe_label_status") == "unreviewed" and not allow_unreviewed_bigvul:
                counts["skipped_unreviewed_bigvul"] += 1
                continue
            if row.get("cwe_multilabel"):
                counts["skipped_multilabel"] += 1
                continue
            vulnerable = bool(row.get("is_vulnerable"))
            cwe_id = row.get("cwe_id") if vulnerable else None
            if vulnerable and not cwe_id:
                counts["skipped_missing_cwe"] += 1
                continue
            assistant = {
                "verdict": "vulnerable" if vulnerable else "safe",
                "cwe_id": cwe_id,
                "evidence_node_ids": [],
            }
            messages = [
                {"role": "system", "content": SINGLE_TURN_SYSTEM_PROMPT},
                {"role": "user", "content": "Function source:\n```c/c++\n" + str(row["code"]) + "\n```"},
                {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False, sort_keys=True)},
            ]
            handle.write(json.dumps({"sample_id": row["sample_id"], "messages": messages}, ensure_ascii=False) + "\n")
            counts["written"] += 1
    counts["out"] = str(output)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Normalized training JSONL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-unreviewed-bigvul-cwe", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.out, args.allow_unreviewed_bigvul_cwe), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
