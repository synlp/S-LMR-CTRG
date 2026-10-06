from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .slmr import SLMRConfig, load_decomposition


@dataclass(frozen=True)
class ReportSample:
    sample_id: str
    image_path: Path
    report: str
    order_key: str | None = None
    layout: str = "auto"


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def ordered_slice_files(path: str | Path) -> list[Path]:
    root = Path(path)
    suffixes = {".png", ".jpg", ".jpeg"}
    return sorted((item for item in root.iterdir() if item.suffix.lower() in suffixes), key=natural_key)


def apply_slice_order(volume: np.ndarray, order: list[int] | None) -> np.ndarray:
    if order is None:
        return volume
    if sorted(order) != list(range(len(order))):
        raise ValueError("slice order must be a permutation")
    if len(order) != volume.shape[0]:
        raise ValueError("slice order length must match the volume")
    return volume[np.asarray(order, dtype=np.int64)]


def to_grayscale(volume: np.ndarray, layout: str = "auto") -> np.ndarray:
    array = np.asarray(volume)
    if layout == "CDHW" and array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("M3D-Cap requires one grayscale channel")
        return array[0]
    if array.ndim == 2:
        return array[None]
    if array.ndim == 3:
        return array
    if array.ndim == 4 and array.shape[-1] in {1, 3}:
        if array.shape[-1] == 1:
            return array[..., 0]
        weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
        return np.tensordot(array, weights, axes=([-1], [0]))
    raise ValueError("volume must have shape [T,H,W] or [T,H,W,C]")


def normalize_preprocessed(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume)
    if not np.isfinite(array).all():
        raise ValueError("volume contains non-finite values")
    if np.issubdtype(array.dtype, np.integer):
        if array.min(initial=0) < 0:
            raise ValueError("signed integer volumes must be preprocessed to [0,1]")
        maximum = float(np.iinfo(array.dtype).max)
        return array.astype(np.float32) / maximum
    array = array.astype(np.float32)
    if array.min(initial=0) < 0 or array.max(initial=0) > 1:
        raise ValueError("floating-point volumes must be preprocessed to [0,1]")
    return array


def resize_volume(volume: np.ndarray, image_size: int = 224) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(volume)).float().unsqueeze(1)
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized[:, 0].numpy()


def load_volume(path: str | Path, image_size: int = 224, order: list[int] | None = None, layout: str = "auto") -> np.ndarray:
    source = Path(path)
    if source.is_dir():
        files = ordered_slice_files(source)
        if not files:
            raise FileNotFoundError(f"no image slices found in {source}")
        raw = np.stack([np.asarray(Image.open(item).convert("L")) for item in files])
    elif source.suffix == ".npy":
        raw = np.load(source)
    else:
        raise ValueError("volume path must be an image directory or .npy file")
    volume = to_grayscale(normalize_preprocessed(raw), layout)
    volume = apply_slice_order(volume, order)
    return resize_volume(volume, image_size)


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_order_mapping(path: str | Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    raw = read_json(path)
    mapping: dict[str, list[int]] = {}
    for key, value in raw.items():
        order = value.get("correct_order") if isinstance(value, dict) else value
        mapping[key] = [int(index) for index in order]
    return mapping


def build_m3d_cap_samples(index_path: str | Path, data_root: str | Path, split: str) -> list[ReportSample]:
    raw = read_json(index_path)
    split_name = "validation" if split in {"dev", "valid", "val"} else split
    rows = raw[split_name]
    root = Path(data_root)
    samples = []
    for row in rows:
        image_ref = str(row["image"])
        if image_ref.startswith("M3D_Cap_npy/"):
            relative = image_ref[len("M3D_Cap_npy/") :]
        else:
            relative = image_ref
        order_key = relative[:-4] if relative.endswith(".npy") else relative
        direct = root / image_ref
        image_path = direct if direct.exists() else root / "M3D_Cap" / order_key
        report = str(row.get("report", "")).strip()
        text_ref = row.get("text")
        if not report:
            text_path = root / str(text_ref) if text_ref else root / "M3D_Cap" / str(Path(order_key).parent) / "text.txt"
            if not text_path.exists() and str(text_ref).startswith("M3D_Cap_npy/"):
                text_path = root / "M3D_Cap" / str(text_ref)[len("M3D_Cap_npy/") :]
            report = text_path.read_text(encoding="utf-8").strip()
        if not report:
            raise ValueError(f"empty report for {order_key}")
        sample_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", order_key).strip("_")
        samples.append(ReportSample(sample_id, image_path, report, order_key, "CDHW"))
    return samples








class CTReportDataset(Dataset):
    def __init__(
        self,
        samples: list[ReportSample],
        image_size: int,
        decomposition_root: str | Path | None = None,
        order_mapping: dict[str, list[int]] | None = None,
        slmr_config: SLMRConfig | None = None,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.decomposition_root = Path(decomposition_root) if decomposition_root else None
        self.order_mapping = order_mapping or {}
        self.slmr_config = slmr_config or SLMRConfig()
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError("duplicate sample ids would share decomposition caches")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        order = self.order_mapping.get(sample.order_key or "")
        original = load_volume(sample.image_path, self.image_size, order, sample.layout)
        item: dict[str, Any] = {"sample_id": sample.sample_id, "original": original, "report": sample.report}
        if self.decomposition_root is not None:
            decomposition = load_decomposition(self.decomposition_root / f"{sample.sample_id}.npz", original, self.slmr_config)
            item["sparse"] = decomposition["sparse"]
            item["low_rank"] = decomposition["low_rank"]
        return item
