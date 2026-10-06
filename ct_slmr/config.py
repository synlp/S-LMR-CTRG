from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: dict[str, Any]
    model: dict[str, Any]
    slmr: dict[str, Any]
    training: dict[str, Any]
    generation: dict[str, Any]
    runtime: dict[str, Any]

    def validate(self) -> None:
        if self.dataset.get("name") != "m3d_cap":
            raise ValueError("dataset.name must be m3d_cap")
        if self.model.get("name_or_path") != "Qwen/Qwen2-VL-2B-Instruct":
            raise ValueError("model.name_or_path must match the paper backbone")
        if int(self.model.get("image_size", 0)) != 224:
            raise ValueError("model.image_size must be 224")
        if self.model.get("streams") != ["sparse", "original", "low_rank"]:
            raise ValueError("model.streams must follow Equation 8")
        expected_slmr = {"lambda": 1.0, "mu": 10.0, "max_iter": 3000, "tolerance": 1e-4}
        for key, value in expected_slmr.items():
            if float(self.slmr.get(key, -1)) != value:
                raise ValueError(f"slmr.{key} must match the paper")
        expected_training = {"epochs": 3, "learning_rate": 5e-5, "batch_size": 16, "weight_decay": 0.01}
        for key, value in expected_training.items():
            if float(self.training.get(key, -1)) != value:
                raise ValueError(f"training.{key} must match the paper")
        if self.generation.get("strategy") != "greedy":
            raise ValueError("generation.strategy must be greedy")


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream)
    config = ExperimentConfig(
        dataset=raw["dataset"],
        model=raw["model"],
        slmr=raw["slmr"],
        training=raw["training"],
        generation=raw["generation"],
        runtime=raw.get("runtime", {}),
    )
    config.validate()
    return config
