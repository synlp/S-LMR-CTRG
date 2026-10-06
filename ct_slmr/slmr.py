from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SLMRConfig:
    lambda_value: float = 1.0
    mu: float = 10.0
    max_iter: int = 3000
    tolerance: float = 1e-4

    def validate(self) -> None:
        if not all(math.isfinite(value) for value in (self.lambda_value, self.mu, self.tolerance)):
            raise ValueError("SLMR settings must be finite")
        if self.lambda_value < 0:
            raise ValueError("lambda_value must be nonnegative")
        if self.mu <= 0:
            raise ValueError("mu must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


@dataclass(frozen=True)
class SLMRResult:
    low_rank: torch.Tensor
    sparse: torch.Tensor
    multipliers: torch.Tensor
    iterations: int
    relative_change: float


def soft_threshold(matrix: torch.Tensor, threshold: float) -> torch.Tensor:
    return matrix.sign() * (matrix.abs() - threshold).clamp_min(0)


def singular_value_threshold(matrix: torch.Tensor, threshold: float) -> torch.Tensor:
    left, singular_values, right = torch.linalg.svd(matrix, full_matrices=False)
    retained = (singular_values - threshold).clamp_min(0)
    return (left * retained.unsqueeze(0)) @ right


def initialize_variables(slices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    low_rank = torch.zeros_like(slices[0])
    sparse = torch.zeros_like(slices)
    multipliers = torch.zeros_like(slices)
    return low_rank, sparse, multipliers


def shared_low_rank_matrix_recovery(
    slices: torch.Tensor,
    config: SLMRConfig | None = None,
) -> SLMRResult:
    cfg = config or SLMRConfig()
    cfg.validate()
    if slices.ndim != 3:
        raise ValueError("slices must have shape [T, H, W]")
    if any(size == 0 for size in slices.shape):
        raise ValueError("slices must contain at least one matrix")
    if not torch.isfinite(slices).all():
        raise ValueError("slices must contain only finite values")
    if not slices.is_floating_point():
        slices = slices.float()
    low_rank, sparse, multipliers = initialize_variables(slices)
    relative_change = float("inf")
    iterations = 0
    for iteration in range(cfg.max_iter):
        previous_low_rank = low_rank
        sparse = soft_threshold(
            slices - previous_low_rank.unsqueeze(0) + multipliers / cfg.mu,
            cfg.lambda_value / cfg.mu,
        )
        average = (slices - sparse + multipliers / cfg.mu).mean(dim=0)
        low_rank = singular_value_threshold(average, 1.0 / cfg.mu)
        multipliers = multipliers + cfg.mu * (slices - low_rank.unsqueeze(0) - sparse)
        numerator = torch.linalg.vector_norm(low_rank - previous_low_rank)
        denominator = torch.linalg.vector_norm(previous_low_rank) + 1e-8
        relative_change = float((numerator / denominator).detach().cpu())
        iterations = iteration + 1
        if relative_change < cfg.tolerance:
            break
    return SLMRResult(low_rank, sparse, multipliers, iterations, relative_change)


def volume_fingerprint(volume: np.ndarray | torch.Tensor) -> str:
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    array = np.ascontiguousarray(volume, dtype="<f4")
    digest = hashlib.sha256(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def save_decomposition(path: str | Path, result: SLMRResult, source: np.ndarray | torch.Tensor | None = None, config: SLMRConfig | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        low_rank=result.low_rank.detach().cpu().numpy().astype(np.float32),
        sparse=result.sparse.detach().cpu().numpy().astype(np.float32),
        iterations=np.asarray(result.iterations, dtype=np.int64),
        relative_change=np.asarray(result.relative_change, dtype=np.float64),
        source_sha256=np.asarray(volume_fingerprint(source) if source is not None else ""),
        config=np.asarray(json.dumps(asdict(config or SLMRConfig()), sort_keys=True)),
    )


def load_decomposition(path: str | Path, source: np.ndarray | None = None, config: SLMRConfig | None = None) -> dict[str, np.ndarray | int | float]:
    with np.load(path, allow_pickle=False) as archive:
        result = {
            "low_rank": archive["low_rank"],
            "sparse": archive["sparse"],
            "iterations": int(archive["iterations"]),
            "relative_change": float(archive["relative_change"]),
        }
        if source is not None:
            if "source_sha256" not in archive or str(archive["source_sha256"]) != volume_fingerprint(source):
                raise ValueError("decomposition does not match this volume and slice order; rerun decompose")
            if result["sparse"].shape != source.shape or result["low_rank"].shape != source.shape[1:]:
                raise ValueError("decomposition shape mismatch")
        if config is not None:
            if "config" not in archive or json.loads(str(archive["config"])) != asdict(config):
                raise ValueError("decomposition settings changed; rerun decompose")
    if not np.isfinite(result["sparse"]).all() or not np.isfinite(result["low_rank"]).all():
        raise ValueError("decomposition contains non-finite values")
    return result
