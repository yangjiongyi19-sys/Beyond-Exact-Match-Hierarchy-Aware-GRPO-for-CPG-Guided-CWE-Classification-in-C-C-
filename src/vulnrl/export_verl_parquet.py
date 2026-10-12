"""Export normalized JSONL rows to VERL's prompt-keyed Parquet input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .prepare_data import _read_records


SYSTEM_MESSAGE = (
    "Classify a C/C++ function as safe or vulnerable. Use bounded read-only CPG tools; "
    "do not use labels, CVE descriptions, or patch metadata. Cite only observed CPG node IDs."
)


def export(input_path: str | Path, output_path: str | Path, allow_unreviewed_bigvul: bool = False) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install the data extra and pyarrow to export VERL Parquet files") from exc

    records = _read_records(input_path)
    rows = []
    excluded = {"unreviewed_bigvul_cwe": 0, "missing_vulnerable_cwe": 0}
    for record in records:
        if (
            record.get("source") == "bigvul"
            and record.get("cwe_label_status") == "unreviewed"
            and not allow_unreviewed_bigvul
        ):
            excluded["unreviewed_bigvul_cwe"] += 1
            continue
        if record.get("is_vulnerable") and not record.get("cwe_id"):
            excluded["missing_vulnerable_cwe"] += 1
            continue
        prompt = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": "Analyze this C/C++ function:\n```c/c++\n" + record["code"] + "\n```"},
        ]
        rows.append({
            "data_source": record.get("source", "unknown"),
            "prompt": prompt,
            "sample_id": str(record["sample_id"]),
            "uid": str(record["sample_id"]),
            "code": str(record["code"]),
            "is_vulnerable": bool(record["is_vulnerable"]),
            "cwe_id": record.get("cwe_id"),
            "project": record.get("project"),
            "source": record.get("source"),
            "cwe_label_status": record.get("cwe_label_status"),
            "reward_model": {"style": "rule", "ground_truth": "labels withheld from prompt"},
        })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    return {"input_rows": len(records), "written_rows": len(rows), "excluded": excluded, "out": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Normalized JSONL, usually PrimeVul train or train_augmented")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-unreviewed-bigvul-cwe", action="store_true")
    args = parser.parse_args()
    report = export(args.input, args.out, args.allow_unreviewed_bigvul_cwe)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

