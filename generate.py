from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ct_slmr.config import load_config
from ct_slmr.pipeline import build_dataset, load_checkpoint
from ct_slmr.qwen2vl import QwenReportProcessor, SLMRQwen2VL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--index-path")
    parser.add_argument("--order-path")
    parser.add_argument("--decomposition-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split = config.dataset["eval_split"]
    dataset = build_dataset(
        config,
        split,
        args.data_root,
        args.index_path,
        args.order_path,
        Path(args.decomposition_root) / split,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=list)
    processor = QwenReportProcessor(
        config.model["name_or_path"],
        int(config.runtime["max_target_length"]),
        args.local_files_only,
    )
    model = SLMRQwen2VL(
        config.model["name_or_path"],
        str(config.runtime["torch_dtype"]),
        args.local_files_only,
    )
    load_checkpoint(args.checkpoint, model)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for batch in tqdm(loader, desc=f"generate:{split}"):
            item = batch[0]
            prediction = model.generate(
                item,
                processor,
                config.model["prompt"],
                args.max_new_tokens,
            )
            row = {"sample_id": item["sample_id"], "prediction": prediction, "reference": item["report"]}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
