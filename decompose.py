from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ct_slmr.config import load_config
from ct_slmr.pipeline import build_dataset
from ct_slmr.slmr import SLMRConfig, load_decomposition, save_decomposition, shared_low_rank_matrix_recovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split")
    parser.add_argument("--data-root")
    parser.add_argument("--index-path")
    parser.add_argument("--order-path")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split = args.split or config.dataset["train_split"]
    dataset = build_dataset(config, split, args.data_root, args.index_path, args.order_path, None)
    slmr_config = SLMRConfig(
        float(config.slmr["lambda"]),
        float(config.slmr["mu"]),
        int(config.slmr["max_iter"]),
        float(config.slmr["tolerance"]),
    )
    output_root = Path(args.output_root) / split
    completed = []
    count = len(dataset)
    for index in tqdm(range(count), desc=f"decompose:{split}"):
        item = dataset[index]
        target = output_root / f"{item['sample_id']}.npz"
        if target.exists() and not args.overwrite:
            try:
                load_decomposition(target, item["original"], slmr_config)
                continue
            except ValueError:
                pass
        slices = torch.from_numpy(item["original"]).to(args.device)
        result = shared_low_rank_matrix_recovery(slices, slmr_config)
        save_decomposition(target, result, item["original"], slmr_config)
        completed.append(
            {
                "sample_id": item["sample_id"],
                "iterations": result.iterations,
                "relative_change": result.relative_change,
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "decomposition.json").open("w", encoding="utf-8") as stream:
        json.dump(completed, stream, indent=2)


if __name__ == "__main__":
    main()
