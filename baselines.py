from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from config import GenerationConfig, MetricConfig, ModelConfig, TrainConfig
from data_pipeline import build_centralized_dataloaders
from modeling import (
    build_inference_model_config,
    evaluate_cross_entropy_loss,
    evaluate_generation_metrics,
    load_base_model_and_tokenizer,
    place_model_on_device,
    run_local_training,
    save_adapter_npz,
    setup_hpc_runtime,
    setup_lora_model,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _cleanup_model(model: Optional[torch.nn.Module]) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_zero_shot_baseline(
    *,
    data_dir: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    generation_config: GenerationConfig,
    metric_config: MetricConfig,
    eval_batch_size: int,
    num_workers: int,
    max_eval_examples: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_config = build_inference_model_config(model_config)
    model, tokenizer = load_base_model_and_tokenizer(inference_config)
    place_model_on_device(model, device)
    loaders = build_centralized_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        train_config=train_config,
        train_batch_size=1,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        max_eval_examples=max_eval_examples,
        seed=seed,
    )

    start = time.time()
    test_loss = evaluate_cross_entropy_loss(model, loaders["test_loss"], device)
    test_generation = evaluate_generation_metrics(
        model=model,
        tokenizer=tokenizer,
        dataloader=loaders["test_gen"],
        generation_config=generation_config,
        device=device,
        metric_config=metric_config,
    )
    total_seconds = time.time() - start
    _cleanup_model(model)
    return {
        "baseline": "zero_shot_llama",
        "total_seconds": total_seconds,
        "test": {**test_loss, **test_generation},
    }


def run_centralized_lora_baseline(
    *,
    data_dir: Path,
    output_dir: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    generation_config: GenerationConfig,
    metric_config: MetricConfig,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    max_train_examples: Optional[int],
    max_eval_examples: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = setup_lora_model(model_config)
    place_model_on_device(model, device)
    loaders = build_centralized_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        train_config=train_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        max_train_examples=max_train_examples,
        max_eval_examples=max_eval_examples,
        seed=seed,
    )

    start = time.time()
    train_metrics = run_local_training(
        model=model,
        train_dataloader=loaders["train"],
        train_config=train_config,
        device=device,
    )
    validation_loss = evaluate_cross_entropy_loss(model, loaders["val_loss"], device)
    test_loss = evaluate_cross_entropy_loss(model, loaders["test_loss"], device)
    test_generation = evaluate_generation_metrics(
        model=model,
        tokenizer=tokenizer,
        dataloader=loaders["test_gen"],
        generation_config=generation_config,
        device=device,
        metric_config=metric_config,
    )
    total_seconds = time.time() - start
    save_adapter_npz(model, output_dir / "centralized_lora_adapter.npz")
    _cleanup_model(model)
    return {
        "baseline": "centralized_lora",
        "total_seconds": total_seconds,
        "train": train_metrics,
        "validation": validation_loss,
        "test": {**test_loss, **test_generation},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run proposal baselines: zero-shot Llama-3.1 and centralized LoRA."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Prepared data directory from data_pipeline.py.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["zero_shot", "centralized_lora", "both"], default="both")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--bertscore-model-type", type=str, default=None)
    parser.add_argument("--bertscore-batch-size", type=int, default=16)

    parser.add_argument("--model-name", type=str, default=ModelConfig.model_name)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)

    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=10)

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path("results") / "baselines"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_hpc_runtime(args.seed)

    model_config = ModelConfig(
        model_name=args.model_name,
        cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir else None,
    )
    train_config = TrainConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        logging_steps=args.logging_steps,
    )
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    metric_config = MetricConfig(
        compute_bertscore=not args.skip_bertscore,
        bertscore_model_type=args.bertscore_model_type,
        bertscore_batch_size=args.bertscore_batch_size,
    )

    results: Dict[str, Any] = {"config": vars(args)}
    if args.mode in {"zero_shot", "both"}:
        print("[baseline] running zero-shot Llama-3.1")
        results["zero_shot"] = run_zero_shot_baseline(
            data_dir=args.data_dir,
            model_config=model_config,
            train_config=train_config,
            generation_config=generation_config,
            metric_config=metric_config,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            max_eval_examples=args.max_eval_examples,
            seed=args.seed,
        )
    if args.mode in {"centralized_lora", "both"}:
        print("[baseline] running centralized LoRA")
        results["centralized_lora"] = run_centralized_lora_baseline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_config=model_config,
            train_config=train_config,
            generation_config=generation_config,
            metric_config=metric_config,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            max_train_examples=args.max_train_examples,
            max_eval_examples=args.max_eval_examples,
            seed=args.seed,
        )

    out_file = args.output_dir / "baseline_results.json"
    with out_file.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(results), handle, indent=2, ensure_ascii=False)
    print(f"[baseline] saved results to {out_file}")


if __name__ == "__main__":
    main()
