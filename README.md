# Feature Decomposition via Shared Low-Rank Matrix Recovery for CT Report Generation

This repository contains the implementation of [Feature Decomposition via Shared Low-Rank Matrix Recovery for CT Report Generation](https://doi.org/10.1109/TMI.2025.3628159), published in IEEE Transactions on Medical Imaging, 45(4):1501–1512, 2026.

## Requirements

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[evaluation]'
```

The backbone is [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct). Transformers loads its model and processor resources. Use `--local-files-only` when those resources are already in the Hugging Face cache. NLG evaluation uses NLTK WordNet resources and a BERTScore checkpoint.

## Data

The included configuration runs M3D-Cap.

- **M3D-Cap:** obtain the dataset and preparation instructions from its original [ModelScope distribution](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Cap) or the authors' [Hugging Face distribution](https://huggingface.co/datasets/GoodBaiBai88/M3D-Cap). Retain `M3D_Cap/M3D_Cap.json` and its `train`, `validation`, and `test` partitions. Place the dataset under `data/M3D-Cap`, or set `dataset.data_root` and `dataset.index_path` in `configs/m3d_cap.json`. Index records point to `image` volumes and `text` report files. The loader supports the prepared one-channel `[1,D,H,W]` NumPy volumes and the corresponding grayscale PNG folders under `M3D_Cap/`.
- **CT-RATE:** request access and accept the conditions at the [official distribution](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE). Obtain the train/validation CT volumes and report CSVs. Convert NIfTI volumes outside this package to ordered axial `[D,H,W]` NumPy arrays in `[0,1]`, retain each `VolumeName`, and pair it with `Findings_EN`. Preserve the original patient partition; the paper evaluates the validation partition. This distribution documents CT-RATE preparation, while the included configuration and entry points use M3D-Cap.

Integer slices are scaled by their dtype range; floating volumes must already be in `[0,1]`. Every slice is retained and resized to `224 × 224`. Supply physical slice-order corrections through `--order-path`, or place `slice_order_mapping.json` under the dataset root. Mapping keys remove the `M3D_Cap_npy/` prefix and `.npy` suffix; each value is a complete index permutation or an object containing `correct_order`. Use the same mapping for decomposition, training, and generation. All stages verify caches against the exact preprocessed slice sequence.

## Run

```bash
python decompose.py --config configs/m3d_cap.json --split train --output-root outputs/decompositions --device cuda
python decompose.py --config configs/m3d_cap.json --split test --output-root outputs/decompositions --device cuda
python train.py --config configs/m3d_cap.json --decomposition-root outputs/decompositions --output-dir outputs/model
python generate.py --config configs/m3d_cap.json --checkpoint outputs/model/checkpoint.pt --decomposition-root outputs/decompositions --output outputs/predictions.jsonl
python evaluate.py --predictions outputs/predictions.jsonl --output outputs/metrics.json
```

S-LMR uses zero initialization, `K=3000`, `lambda=1`, `mu=10`, and tolerance `1e-4`. Sparse and original slices use independent recurrent visual encoders; the shared background uses a third encoder. Their projected features are concatenated in sparse, original, background, prompt order. Signed components use a volume-level symmetric float mapping into the Qwen image processor's range. Training updates all parameters for three epochs with batch size 16, learning rate `5e-5`, and weight decay `0.01`; generation uses greedy decoding.

## Evaluation

The evaluator reports BLEU-1 through BLEU-4, ROUGE-1, ROUGE-L, METEOR, and BERTScore. Install WordNet resources before METEOR evaluation:

```bash
python -m nltk.downloader wordnet omw-1.4
```

For the M3D-Cap clinical metric, install the [official RaTEScore implementation](https://github.com/MAGIC-AI4Med/RaTEScore) separately. Pass `--ratescore-command` a command that reads `{input}` JSONL records containing `prediction` and `reference` and writes `{output}` as a JSON object of numeric scores.

## Structure

```text
CT-S-LMR/
├── README.md
├── pyproject.toml
├── configs/
│   └── m3d_cap.json
├── ct_slmr/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── qwen2vl.py
│   └── slmr.py
├── decompose.py
├── train.py
├── generate.py
└── evaluate.py
```

`ct_slmr/slmr.py` implements Algorithm 1; `ct_slmr/qwen2vl.py` implements continuous slice encoding and report generation. The other modules in `ct_slmr/` handle CT data, settings, checkpoint operations, and metrics. `configs/m3d_cap.json` contains the run configuration. Run the four root-level entry scripts directly.

## Citation

If you use this code, please cite the paper.

```bibtex
@article{tian2026feature,
  title={Feature Decomposition via Shared Low-Rank Matrix Recovery for CT Report Generation},
  author={Tian, Yuanhe and Song, Yan},
  journal={IEEE Transactions on Medical Imaging},
  volume={45},
  number={4},
  pages={1501--1512},
  year={2026}
}
```
