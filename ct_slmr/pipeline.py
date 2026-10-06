from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .data import CTReportDataset, build_m3d_cap_samples, read_order_mapping


def build_dataset(config: ExperimentConfig, split: str, data_root: str | Path | None, index_path: str | Path | None, order_path: str | Path | None, decomposition_root: str | Path | None) -> CTReportDataset:
    dataset = config.dataset
    root = Path(data_root or dataset["data_root"])
    index = Path(index_path or dataset["index_path"])
    samples = build_m3d_cap_samples(index, root, split)
    order_source = order_path or dataset.get("order_path")
    if order_source is None and (root / "slice_order_mapping.json").is_file():
        order_source = root / "slice_order_mapping.json"
    order = read_order_mapping(order_source)
    return CTReportDataset(samples, int(config.model["image_size"]), decomposition_root, order)


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, step: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        },
        target,
    )


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
