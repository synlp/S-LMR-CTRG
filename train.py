from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ct_slmr.config import load_config
from ct_slmr.pipeline import build_dataset, load_checkpoint, save_checkpoint
from ct_slmr.qwen2vl import QwenReportProcessor, SLMRQwen2VL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--index-path")
    parser.add_argument("--order-path")
    parser.add_argument("--decomposition-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split = config.dataset["train_split"]
    decomposition_root = Path(args.decomposition_root) / split
    dataset = build_dataset(
        config,
        split,
        args.data_root,
        args.index_path,
        args.order_path,
        decomposition_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.training["batch_size"]),
        shuffle=True,
        num_workers=int(config.runtime.get("num_workers", 0)),
        collate_fn=list,
    )
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
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["step"])
    model.train()
    losses = []
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, int(config.training["epochs"])):
        last_epoch = epoch
        progress = tqdm(loader, desc=f"train:{epoch + 1}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, processor, config.model["prompt"])
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            global_step += 1
            value = float(loss.detach().cpu())
            losses.append(value)
            progress.set_postfix(loss=f"{value:.4f}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output_dir / "checkpoint.pt", model, optimizer, last_epoch, global_step)
    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump({"steps": global_step, "losses": losses}, stream)


if __name__ == "__main__":
    main()
