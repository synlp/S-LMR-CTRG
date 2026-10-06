from __future__ import annotations

import argparse
import json
from pathlib import Path

from ct_slmr.metrics import compute_nlg_metrics, run_external_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratescore-command")
    parser.add_argument("--green-command")
    parser.add_argument("--clinical-f1-command")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.predictions).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    predictions = [row["prediction"] for row in rows]
    references = [row["reference"] for row in rows]
    metrics = compute_nlg_metrics(predictions, references)
    for name, command in [
        ("ratescore", args.ratescore_command),
        ("green", args.green_command),
        ("clinical_f1", args.clinical_f1_command),
    ]:
        if command:
            result = run_external_metric(command, rows)
            metrics.update({f"{name}_{key}": value for key, value in result.items()})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)


if __name__ == "__main__":
    main()
