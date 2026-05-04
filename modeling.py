from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from config import GenerationConfig, MetricConfig, ModelConfig, TrainConfig


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def setup_hpc_runtime(seed: Optional[int] = None) -> None:
    """Configure fast, non-deterministic CUDA defaults that fit H100/H200 jobs."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_torch_dtype(preferred: str) -> torch.dtype:
    if preferred not in {"bfloat16", "float16", "float32"}:
        raise ValueError("preferred must be bfloat16, float16, or float32")
    if not torch.cuda.is_available():
        return torch.float32
    if preferred == "bfloat16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if preferred == "float16":
        return torch.float16
    return torch.float32


def _enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    base = getattr(model, "base_model", None)
    inner = getattr(base, "model", None)
    for candidate in (model, base, inner):
        hook = getattr(candidate, "_require_grads_hook", None) if candidate is not None else None
        if hook is not None:
            try:
                hook.remove()
            except RuntimeError:
                pass
            candidate._require_grads_hook = None

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()


def _gradient_checkpointing_is_enabled(model: torch.nn.Module) -> bool:
    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if bool(getattr(candidate, "is_gradient_checkpointing", False)):
            return True
    return False


def _disable_gradient_checkpointing_for_eval(model: torch.nn.Module) -> bool:
    was_enabled = _gradient_checkpointing_is_enabled(model)
    if was_enabled and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return was_enabled


def _restore_gradient_checkpointing(model: torch.nn.Module, was_enabled: bool) -> None:
    if was_enabled:
        _enable_gradient_checkpointing(model)


def load_base_model_and_tokenizer(
    model_config: ModelConfig,
) -> Tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    dtype = resolve_torch_dtype(model_config.preferred_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        cache_dir=model_config.cache_dir,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "cache_dir": model_config.cache_dir,
        "low_cpu_mem_usage": model_config.low_cpu_mem_usage,
    }
    if model_config.attn_implementation:
        model_kwargs["attn_implementation"] = model_config.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_config.model_name, **model_kwargs)
    model.config.use_cache = model_config.use_cache_during_training
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model_config.use_gradient_checkpointing:
        _enable_gradient_checkpointing(model)
    return model, tokenizer


def setup_lora_model(
    model_config: ModelConfig,
) -> Tuple[PeftModel, PreTrainedTokenizerBase]:
    base_model, tokenizer = load_base_model_and_tokenizer(model_config)
    lora_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=list(model_config.target_modules),
        bias=model_config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    print_trainable_parameters(model)
    return model, tokenizer


def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    pct = 100.0 * trainable / total if total else 0.0
    print(
        f"[model] trainable params: {trainable:,} | all params: {total:,} | trainable%: {pct:.4f}"
    )


def place_model_on_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Move the model to the active device."""
    return model.to(device)


def generation_stop_token_ids(tokenizer: Any) -> List[int]:
    """Return EOS ids for Llama chat generation, including <|eot_id|>."""
    stop_ids: List[int] = []
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for token_id in (
        getattr(tokenizer, "eos_token_id", None),
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if hasattr(tokenizer, "convert_tokens_to_ids")
        else None,
    ):
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
            if token_id not in stop_ids:
                stop_ids.append(token_id)
    return stop_ids


# ---------------------------------------------------------------------------
# Adapter-only state transport
# ---------------------------------------------------------------------------
def get_lora_parameter_names(model: torch.nn.Module) -> List[str]:
    return sorted(get_peft_model_state_dict(model).keys())


def get_lora_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    state = get_peft_model_state_dict(model)
    arrays: List[np.ndarray] = []
    for name in sorted(state.keys()):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Adapter state {name!r} is not a torch.Tensor")
        arrays.append(tensor.detach().to(torch.float32).cpu().numpy())
    return arrays


def set_lora_parameters(model: torch.nn.Module, parameters: Sequence[np.ndarray]) -> None:
    current_state = get_peft_model_state_dict(model)
    names = sorted(current_state.keys())
    if len(parameters) != len(names):
        raise ValueError(f"Received {len(parameters)} arrays but model expects {len(names)}")

    new_state: Dict[str, torch.Tensor] = {}
    for name, array in zip(names, parameters):
        target = current_state[name]
        if tuple(array.shape) != tuple(target.shape):
            raise ValueError(
                f"Shape mismatch for {name}: received {tuple(array.shape)}, expected {tuple(target.shape)}"
            )
        new_state[name] = torch.from_numpy(np.asarray(array)).to(
            dtype=target.dtype,
            device=target.device,
        )
    set_peft_model_state_dict(model, new_state)


def adapter_size_bytes_from_arrays(arrays: Sequence[np.ndarray]) -> int:
    return int(sum(array.nbytes for array in arrays))


def adapter_size_mb_from_arrays(arrays: Sequence[np.ndarray]) -> float:
    return adapter_size_bytes_from_arrays(arrays) / (1024**2)


def save_adapter_npz(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = get_lora_parameters(model)
    names = np.array(get_lora_parameter_names(model), dtype=object)
    payload = {f"arr_{index:04d}": array for index, array in enumerate(arrays)}
    payload["names"] = names
    np.savez_compressed(path, **payload)


# ---------------------------------------------------------------------------
# Training and loss evaluation
# ---------------------------------------------------------------------------
def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


def count_examples(dataloader: DataLoader) -> int:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None:
        try:
            return int(len(dataset))
        except TypeError:
            pass
    try:
        return int(len(dataloader))
    except TypeError:
        return 0


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_config: TrainConfig,
) -> Dict[str, float]:
    model.train()
    place_model_on_device(model, device)
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_batches = 0
    for step, batch in enumerate(dataloader):
        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
        )
        loss = outputs.loss
        (loss / train_config.gradient_accumulation_steps).backward()
        total_loss += float(loss.detach().item())
        total_batches += 1

        is_accumulation_step = (step + 1) % train_config.gradient_accumulation_steps == 0
        is_last_step = (step + 1) == len(dataloader)
        if is_accumulation_step or is_last_step:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=train_config.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if train_config.logging_steps > 0 and (step + 1) % train_config.logging_steps == 0:
            print(f"[train] step={step + 1} avg_loss={total_loss / max(1, total_batches):.4f}")

    return {"train_loss": total_loss / max(1, total_batches)}


def run_local_training(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    train_config: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    metrics: Dict[str, float] = {"train_loss": float("nan")}
    start = time.time()
    for epoch in range(train_config.num_epochs):
        print(f"[train] epoch {epoch + 1}/{train_config.num_epochs}")
        metrics = train_one_epoch(model, train_dataloader, optimizer, device, train_config)
    metrics["train_seconds"] = time.time() - start
    metrics["num_epochs"] = float(train_config.num_epochs)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return metrics


@torch.no_grad()
def evaluate_cross_entropy_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    place_model_on_device(model, device)
    was_gradient_checkpointing = _disable_gradient_checkpointing_for_eval(model)
    total_token_loss = 0.0
    total_tokens = 0
    start = time.time()

    try:
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            labels = batch["labels"]
            token_count = int((labels != -100).sum().item())
            if token_count == 0:
                continue
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=labels,
            )
            total_token_loss += float(outputs.loss.detach().item()) * token_count
            total_tokens += token_count
    finally:
        _restore_gradient_checkpointing(model, was_gradient_checkpointing)

    loss = total_token_loss / total_tokens if total_tokens else 0.0
    return {
        "eval_loss": float(loss),
        "eval_perplexity": float(math.exp(min(loss, 20.0))) if total_tokens else 0.0,
        "num_loss_tokens": float(total_tokens),
        "num_eval_examples": float(count_examples(dataloader)),
        "loss_eval_seconds": time.time() - start,
    }


# ---------------------------------------------------------------------------
# Generation and metrics
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_metric(metric_name: str) -> Any:
    import evaluate

    return evaluate.load(metric_name)


def _clean_metric_text(text: Any) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    return cleaned if cleaned else " "


def compute_summarization_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    metric_config: Optional[MetricConfig] = None,
) -> Dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have equal length, got {len(predictions)} and {len(references)}"
        )
    if not predictions:
        base = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        if metric_config and metric_config.compute_bertscore:
            base.update({"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0})
        return base

    clean_preds = [_clean_metric_text(prediction) for prediction in predictions]
    clean_refs = [_clean_metric_text(reference) for reference in references]

    rouge = _load_metric("rouge")
    rouge_out = rouge.compute(
        predictions=clean_preds,
        references=clean_refs,
        use_stemmer=True,
    )
    metrics = {
        "rouge1": float(rouge_out.get("rouge1", 0.0)),
        "rouge2": float(rouge_out.get("rouge2", 0.0)),
        "rougeL": float(rouge_out.get("rougeL", 0.0)),
    }

    metric_config = metric_config or MetricConfig(compute_bertscore=False)
    if metric_config.compute_bertscore:
        bertscore = _load_metric("bertscore")
        kwargs: Dict[str, Any] = {
            "predictions": clean_preds,
            "references": clean_refs,
            "lang": metric_config.bertscore_lang,
            "batch_size": metric_config.bertscore_batch_size,
        }
        if metric_config.bertscore_model_type:
            kwargs["model_type"] = metric_config.bertscore_model_type
        bert_out = bertscore.compute(**kwargs)
        metrics.update(
            {
                "bertscore_precision": float(np.mean(bert_out["precision"])),
                "bertscore_recall": float(np.mean(bert_out["recall"])),
                "bertscore_f1": float(np.mean(bert_out["f1"])),
            }
        )
    return metrics


@torch.no_grad()
def generate_summaries(
    model: torch.nn.Module,
    tokenizer: Any,
    dataloader: DataLoader,
    generation_config: GenerationConfig,
    device: torch.device,
) -> Tuple[List[str], List[str]]:
    model.eval()
    place_model_on_device(model, device)
    was_gradient_checkpointing = _disable_gradient_checkpointing_for_eval(model)
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    original_use_cache = getattr(model.config, "use_cache", False)
    tokenizer.padding_side = "left"
    model.config.use_cache = True
    stop_token_ids = generation_stop_token_ids(tokenizer)

    predictions: List[str] = []
    references: List[str] = []
    try:
        for batch in dataloader:
            raw_references = batch.get("reference")
            if raw_references is None:
                raise KeyError("Generation dataloader must provide a reference field")
            batch = move_batch_to_device(batch, device)
            input_ids = batch["input_ids"]
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": batch.get("attention_mask"),
                "max_new_tokens": generation_config.max_new_tokens,
                "num_beams": generation_config.num_beams,
                "do_sample": generation_config.do_sample,
                "repetition_penalty": generation_config.repetition_penalty,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": stop_token_ids or tokenizer.eos_token_id,
            }
            if generation_config.do_sample:
                gen_kwargs["temperature"] = generation_config.temperature
                gen_kwargs["top_p"] = generation_config.top_p

            generated = model.generate(**gen_kwargs)
            new_tokens = generated[:, input_ids.shape[1] :]
            predictions.extend(
                text.strip() for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            )
            references.extend(str(reference).strip() for reference in raw_references)
    finally:
        tokenizer.padding_side = original_padding_side
        model.config.use_cache = original_use_cache
        _restore_gradient_checkpointing(model, was_gradient_checkpointing)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predictions, references


def evaluate_generation_metrics(
    model: torch.nn.Module,
    tokenizer: Any,
    dataloader: DataLoader,
    generation_config: GenerationConfig,
    device: torch.device,
    metric_config: Optional[MetricConfig] = None,
) -> Dict[str, float]:
    start = time.time()
    predictions, references = generate_summaries(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        generation_config=generation_config,
        device=device,
    )
    metrics = compute_summarization_metrics(predictions, references, metric_config)
    metrics["num_eval_examples"] = float(len(references))
    metrics["generation_seconds"] = time.time() - start
    return metrics


def build_inference_model_config(model_config: ModelConfig) -> ModelConfig:
    return replace(
        model_config,
        use_gradient_checkpointing=False,
        use_cache_during_training=True,
    )
