from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration


def extract_visual_streams(backbone: nn.Module) -> tuple[nn.Module, nn.Module, nn.Module]:
    original = backbone.visual
    sparse = copy.deepcopy(original)
    low_rank = copy.deepcopy(original)
    backbone.visual = nn.Identity()
    return sparse, original, low_rank


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def original_volume(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("original volume must have shape [T,H,W]")
    if not np.isfinite(array).all():
        raise ValueError("original volume contains non-finite values")
    if array.min(initial=0) < 0 or array.max(initial=0) > 1:
        raise ValueError("original volume must be in [0,1]")
    return array


def signed_component_volume(volume: np.ndarray, dimensions: int) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != dimensions:
        raise ValueError(f"signed component must have {dimensions} dimensions")
    if not np.isfinite(array).all():
        raise ValueError("signed component contains non-finite values")
    magnitude = float(np.max(np.abs(array), initial=0))
    if magnitude == 0:
        return np.full_like(array, 0.5)
    return array / (2.0 * magnitude) + 0.5


def prepare_visual_streams(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sparse = signed_component_volume(item["sparse"], 3)
    original = original_volume(item["original"])
    low_rank = signed_component_volume(item["low_rank"], 2)
    if sparse.shape != original.shape:
        raise ValueError("sparse and original volumes must have the same shape")
    if low_rank.shape != original.shape[1:]:
        raise ValueError("low-rank component must match the slice shape")
    return sparse, original, low_rank


def float_rgb_image(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must have shape [H,W]")
    if not np.isfinite(image).all():
        raise ValueError("image contains non-finite values")
    if image.min(initial=0) < 0 or image.max(initial=0) > 1:
        raise ValueError("image must be in [0,1]")
    return np.repeat(image[..., None], 3, axis=-1)


def sequential_mrope_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention mask must have shape [B,L]")
    positions = attention_mask.long().cumsum(dim=-1) - 1
    positions.masked_fill_(attention_mask == 0, 1)
    return positions.unsqueeze(0).expand(3, -1, -1)


class QwenReportProcessor:
    def __init__(self, model_name_or_path: str, max_target_length: int, local_files_only: bool) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_target_length = max_target_length
        if max_target_length < 2:
            raise ValueError("max_target_length must allow content and EOS")

    def image(self, array: np.ndarray) -> dict[str, torch.Tensor]:
        values = self.processor.image_processor(
            images=[float_rgb_image(array)],
            return_tensors="pt",
            do_resize=False,
            do_rescale=False,
        )
        return {"pixel_values": values["pixel_values"], "image_grid_thw": values["image_grid_thw"]}

    def text_ids(self, text: str, add_special_tokens: bool, append_eos: bool) -> torch.Tensor:
        ids = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=self.max_target_length - int(append_eos),
            return_tensors="pt",
        )["input_ids"][0]
        eos = self.tokenizer.eos_token_id
        if append_eos and (ids.numel() == 0 or int(ids[-1]) != eos):
            ids = torch.cat([ids, torch.tensor([eos], dtype=torch.long)])
        return ids


class QwenVisionStream(nn.Module):
    def __init__(self, visual: nn.Module, recurrent: bool) -> None:
        super().__init__()
        self.visual = visual
        self.recurrent = recurrent

    def encode_image(
        self,
        image: dict[str, torch.Tensor],
        previous: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pixel_values = image["pixel_values"].to(device=device, dtype=dtype)
        grid = image["image_grid_thw"].to(device=device)
        hidden = self.visual.patch_embed(pixel_values)
        if self.recurrent and previous is not None:
            if previous.shape != hidden.shape:
                raise ValueError("all recurrent slices must have the same patch shape")
            hidden = hidden + previous
        rotary = self.visual.rot_pos_emb(grid)
        lengths = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(dim=0, dtype=torch.int32)
        lengths = F.pad(lengths, (1, 0), value=0)
        for block in self.visual.blocks:
            hidden = block(hidden, cu_seqlens=lengths, rotary_pos_emb=rotary)
        return hidden

    def forward(
        self,
        images: list[dict[str, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        state = None
        for image in images:
            state = self.encode_image(image, state, device, dtype)
        if state is None:
            raise ValueError("visual stream must contain at least one image")
        return self.visual.merger(state)


class SLMRQwen2VL(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype: str = "bfloat16",
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        dtype = dtype_from_name(torch_dtype)
        self.backbone = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        sparse_visual, original_visual, low_rank_visual = extract_visual_streams(self.backbone)
        self.sparse_encoder = QwenVisionStream(sparse_visual, True)
        self.original_encoder = QwenVisionStream(original_visual, True)
        self.low_rank_encoder = QwenVisionStream(low_rank_visual, False)

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_input_embeddings()(ids)

    def visual_prefix(
        self,
        item: dict[str, Any],
        processor: QwenReportProcessor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sparse_volume, original_volume_data, low_rank_matrix = prepare_visual_streams(item)
        sparse_images = [processor.image(image) for image in sparse_volume]
        original_images = [processor.image(image) for image in original_volume_data]
        low_rank_images = [processor.image(low_rank_matrix)]
        sparse = self.sparse_encoder(sparse_images, device, dtype)
        original = self.original_encoder(original_images, device, dtype)
        low_rank = self.low_rank_encoder(low_rank_images, device, dtype)
        return torch.cat([sparse, original, low_rank], dim=0)

    def sample_inputs(
        self,
        item: dict[str, Any],
        processor: QwenReportProcessor,
        prompt: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.visual_prefix(item, processor, device, dtype)
        prompt_ids = processor.text_ids(prompt, True, False).to(device)
        target_ids = processor.text_ids(item["report"], False, True).to(device)
        embeddings = torch.cat([visual, self.embed_tokens(prompt_ids), self.embed_tokens(target_ids)], dim=0)
        ignored = torch.full((visual.shape[0] + prompt_ids.shape[0],), -100, dtype=torch.long, device=device)
        labels = torch.cat([ignored, target_ids])
        return embeddings, labels

    def forward(
        self,
        batch: list[dict[str, Any]],
        processor: QwenReportProcessor,
        prompt: str,
    ) -> dict[str, torch.Tensor]:
        parameter = next(self.parameters())
        device = parameter.device
        dtype = parameter.dtype
        samples = [self.sample_inputs(item, processor, prompt, device, dtype) for item in batch]
        maximum = max(embeddings.shape[0] for embeddings, _ in samples)
        embeddings_batch = []
        labels_batch = []
        masks = []
        for embeddings, labels in samples:
            padding = maximum - embeddings.shape[0]
            embeddings_batch.append(F.pad(embeddings, (0, 0, 0, padding)))
            labels_batch.append(F.pad(labels, (0, padding), value=-100))
            mask = torch.zeros(maximum, dtype=torch.long, device=device)
            mask[: embeddings.shape[0]] = 1
            masks.append(mask)
        inputs_embeds = torch.stack(embeddings_batch)
        labels = torch.stack(labels_batch)
        attention_mask = torch.stack(masks)
        position_ids = sequential_mrope_position_ids(attention_mask)
        outputs = self.backbone.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = self.backbone.lm_head(outputs.last_hidden_state).float()
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, logits.shape[-1]), shift_labels.view(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        item: dict[str, Any],
        processor: QwenReportProcessor,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        self.eval()
        parameter = next(self.parameters())
        device = parameter.device
        dtype = parameter.dtype
        visual = self.visual_prefix(item, processor, device, dtype)
        prompt_ids = processor.text_ids(prompt, True, False).to(device)
        prefix = torch.cat([visual, self.embed_tokens(prompt_ids)], dim=0)
        generated: list[int] = []
        stop_ids = {processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id}
        for _ in range(max_new_tokens):
            if generated:
                generated_ids = torch.tensor(generated, dtype=torch.long, device=device)
                embeddings = torch.cat([prefix, self.embed_tokens(generated_ids)], dim=0)
            else:
                embeddings = prefix
            mask = torch.ones((1, embeddings.shape[0]), dtype=torch.long, device=device)
            position_ids = sequential_mrope_position_ids(mask)
            outputs = self.backbone.model(
                inputs_embeds=embeddings.unsqueeze(0),
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=False,
            )
            logits = self.backbone.lm_head(outputs.last_hidden_state[:, -1])
            next_id = int(logits.argmax(dim=-1))
            if next_id in stop_ids:
                break
            generated.append(next_id)
        return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
