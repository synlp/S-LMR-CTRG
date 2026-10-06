from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from pathlib import Path


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def compute_nlg_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    from bert_score import score as bert_score
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    references_tokens = [[tokenize(reference)] for reference in references]
    predictions_tokens = [tokenize(prediction) for prediction in predictions]
    metrics = {}
    for order in range(1, 5):
        weights = tuple([1.0 / order] * order + [0.0] * (4 - order))
        metrics[f"bleu_{order}"] = 100 * corpus_bleu(references_tokens, predictions_tokens, weights=weights)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge = [scorer.score(reference, prediction) for prediction, reference in zip(predictions, references)]
    metrics["rouge_1"] = 100 * sum(item["rouge1"].fmeasure for item in rouge) / max(len(rouge), 1)
    metrics["rouge_l"] = 100 * sum(item["rougeL"].fmeasure for item in rouge) / max(len(rouge), 1)
    meteor = [meteor_score([tokenize(reference)], tokenize(prediction)) for prediction, reference in zip(predictions, references)]
    metrics["meteor"] = 100 * sum(meteor) / max(len(meteor), 1)
    _, _, bert_f1 = bert_score(predictions, references, lang="en", verbose=False)
    metrics["bert_score"] = 100 * float(bert_f1.mean())
    return metrics


def run_external_metric(command: str, rows: list[dict[str, str]]) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "predictions.jsonl"
        output_path = root / "metrics.json"
        with input_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        arguments = [part.replace("{input}", str(input_path)).replace("{output}", str(output_path)) for part in shlex.split(command)]
        subprocess.run(arguments, check=True)
        with output_path.open(encoding="utf-8") as stream:
            result = json.load(stream)
    return {str(key): float(value) for key, value in result.items()}
