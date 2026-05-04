from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import Metrics, NDArrays, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader

from config import GenerationConfig, MetricConfig, ModelConfig, TrainConfig
from data_pipeline import build_client_dataloaders, build_centralized_dataloaders, discover_clients
from modeling import (
    adapter_size_mb_from_arrays,
    count_examples,
    evaluate_cross_entropy_loss,
    evaluate_generation_metrics,
    get_lora_parameters,
    place_model_on_device,
    run_local_training,
    save_adapter_npz,
    set_lora_parameters,
    setup_hpc_runtime,
    setup_lora_model,
)


class MedicalFLClient(fl.client.NumPyClient):
    """Flower client that trains locally and exchanges only LoRA adapter arrays."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        train_dataloader: DataLoader,
        val_loss_dataloader: DataLoader,
        val_gen_dataloader: DataLoader,
        client_id: str,
        train_config: TrainConfig,
        generation_config: GenerationConfig,
        generate_during_rounds: bool,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_loss_dataloader = val_loss_dataloader
        self.val_gen_dataloader = val_gen_dataloader
        self.client_id = client_id
        self.train_config = train_config
        self.generation_config = generation_config
        self.generate_during_rounds = generate_during_rounds
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        place_model_on_device(self.model, self.device)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        del config
        return get_lora_parameters(self.model)

    def set_parameters(self, parameters: Sequence[np.ndarray]) -> None:
        set_lora_parameters(self.model, parameters)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", self.train_config.num_epochs))
        train_config = replace(self.train_config, num_epochs=local_epochs)

        start = time.time()
        try:
            train_metrics = run_local_training(
                model=self.model,
                train_dataloader=self.train_dataloader,
                train_config=train_config,
                device=self.device,
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        fit_seconds = time.time() - start

        updated = self.get_parameters(config={})
        adapter_mb = adapter_size_mb_from_arrays(updated)
        metrics = {
            "client_id": self.client_id,
            "fit_seconds": float(fit_seconds),
            "adapter_size_mb": adapter_mb,
            "one_way_comm_mb": adapter_mb,
            "round_trip_comm_mb": 2.0 * adapter_mb,
            **_numeric_metrics(train_metrics),
        }
        return updated, count_examples(self.train_dataloader), metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        del config
        self.set_parameters(parameters)
        start = time.time()
        try:
            loss_metrics = evaluate_cross_entropy_loss(
                model=self.model,
                dataloader=self.val_loss_dataloader,
                device=self.device,
            )
            gen_metrics: Dict[str, float] = {}
            if self.generate_during_rounds:
                gen_metrics = evaluate_generation_metrics(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    dataloader=self.val_gen_dataloader,
                    generation_config=self.generation_config,
                    device=self.device,
                    metric_config=MetricConfig(compute_bertscore=False),
                )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        eval_seconds = time.time() - start

        arrays = self.get_parameters(config={})
        adapter_mb = adapter_size_mb_from_arrays(arrays)
        metrics = {
            "client_id": self.client_id,
            "eval_seconds": float(eval_seconds),
            "adapter_size_mb": adapter_mb,
            "one_way_comm_mb": adapter_mb,
            "round_trip_comm_mb": 2.0 * adapter_mb,
            **_numeric_metrics(loss_metrics),
            **_numeric_metrics(gen_metrics),
        }
        return float(loss_metrics["eval_loss"]), count_examples(self.val_loss_dataloader), metrics


def _numeric_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def _weighted_average(metrics: List[Tuple[int, Dict[str, Any]]], key: str) -> float:
    numerator = 0.0
    denominator = 0
    for num_examples, metric in metrics:
        value = metric.get(key)
        if isinstance(value, (int, float)):
            numerator += float(value) * int(num_examples)
            denominator += int(num_examples)
    return numerator / denominator if denominator else 0.0


def _simple_mean(metrics: List[Tuple[int, Dict[str, Any]]], key: str) -> float:
    values = [float(metric[key]) for _, metric in metrics if isinstance(metric.get(key), (int, float))]
    return sum(values) / len(values) if values else 0.0


def _simple_sum(metrics: List[Tuple[int, Dict[str, Any]]], key: str) -> float:
    return sum(float(metric[key]) for _, metric in metrics if isinstance(metric.get(key), (int, float)))


class RoundLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.records: List[Dict[str, Any]] = []
        self.round_wallclock_seconds: Dict[int, float] = {}
        self._round_start: Dict[int, float] = {}
        self._pending_fit: Dict[Tuple[int, str], Dict[str, Any]] = {}

    def begin_round(self, server_round: int) -> None:
        self._round_start[server_round] = time.time()

    def record_fit(self, server_round: int, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        typed = [(int(num_examples), dict(metric)) for num_examples, metric in metrics]
        for num_examples, metric in typed:
            client_id = str(metric.get("client_id", "unknown"))
            self._pending_fit[(server_round, client_id)] = {
                "round": server_round,
                "client_id": client_id,
                "num_train_examples": int(num_examples),
                "train_loss": float(metric.get("train_loss", 0.0)),
                "fit_seconds": float(metric.get("fit_seconds", 0.0)),
                "adapter_size_mb": float(metric.get("adapter_size_mb", 0.0)),
                "one_way_comm_mb": float(metric.get("one_way_comm_mb", 0.0)),
                "round_trip_comm_mb": float(metric.get("round_trip_comm_mb", 0.0)),
            }
        return {
            "train_loss": _weighted_average(typed, "train_loss"),
            "fit_seconds_mean": _simple_mean(typed, "fit_seconds"),
            "adapter_size_mb_mean": _simple_mean(typed, "adapter_size_mb"),
            "total_one_way_comm_mb": _simple_sum(typed, "one_way_comm_mb"),
            "total_round_trip_comm_mb": _simple_sum(typed, "round_trip_comm_mb"),
            "num_fit_examples": float(sum(num_examples for num_examples, _ in typed)),
            "num_fit_clients": float(len(typed)),
        }

    def record_evaluate(self, server_round: int, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        typed = [(int(num_examples), dict(metric)) for num_examples, metric in metrics]
        if server_round in self._round_start:
            self.round_wallclock_seconds[server_round] = time.time() - self._round_start[server_round]

        for num_examples, metric in typed:
            client_id = str(metric.get("client_id", "unknown"))
            fit_part = self._pending_fit.get((server_round, client_id), {})
            record = {
                **fit_part,
                "round": server_round,
                "client_id": client_id,
                "num_eval_examples": int(num_examples),
                "eval_loss": float(metric.get("eval_loss", 0.0)),
                "eval_perplexity": float(metric.get("eval_perplexity", 0.0)),
                "num_loss_tokens": float(metric.get("num_loss_tokens", 0.0)),
                "eval_seconds": float(metric.get("eval_seconds", 0.0)),
                "round_wallclock_seconds": float(self.round_wallclock_seconds.get(server_round, 0.0)),
            }
            for key in ("rouge1", "rouge2", "rougeL", "generation_seconds"):
                if key in metric:
                    record[key] = float(metric[key])
            self.records.append(record)

        aggregated: Metrics = {
            "eval_loss": _weighted_average(typed, "eval_loss"),
            "eval_perplexity": _weighted_average(typed, "eval_perplexity"),
            "num_eval_examples": float(sum(num_examples for num_examples, _ in typed)),
            "num_eval_clients": float(len(typed)),
        }
        for key in ("rouge1", "rouge2", "rougeL"):
            if any(key in metric for _, metric in typed):
                aggregated[key] = _weighted_average(typed, key)
        return aggregated

    def write(self) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "records": self.records,
            "round_wallclock_seconds": {
                str(key): float(value) for key, value in self.round_wallclock_seconds.items()
            },
            "total_one_way_comm_mb": float(sum(record.get("one_way_comm_mb", 0.0) for record in self.records)),
            "total_round_trip_comm_mb": float(sum(record.get("round_trip_comm_mb", 0.0) for record in self.records)),
        }
        with (self.output_dir / "round_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        if self.records:
            with (self.output_dir / "round_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                fieldnames = sorted({key for record in self.records for key in record})
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.records)
        return summary


class TrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, round_logger: RoundLogger, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.round_logger = round_logger
        self.final_parameters = None

    def configure_fit(self, server_round: int, parameters: Any, client_manager: Any) -> Any:
        self.round_logger.begin_round(server_round)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results: Any, failures: Any) -> Tuple[Any, Metrics]:
        aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
        metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
        return aggregated_parameters, self.round_logger.record_fit(server_round, metrics)

    def aggregate_evaluate(self, server_round: int, results: Any, failures: Any) -> Tuple[Optional[float], Metrics]:
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)
        metrics = [(eval_res.num_examples, eval_res.metrics) for _, eval_res in results]
        return aggregated_loss, self.round_logger.record_evaluate(server_round, metrics)


def fit_config_fn(local_epochs: int) -> Callable[[int], Dict[str, Scalar]]:
    def inner(server_round: int) -> Dict[str, Scalar]:
        return {"local_epochs": int(local_epochs), "server_round": int(server_round)}

    return inner


def _client_index(cid_or_context: Any) -> int:
    if isinstance(cid_or_context, str):
        return int(cid_or_context)
    node_config = getattr(cid_or_context, "node_config", None)
    if isinstance(node_config, dict):
        for key in ("partition-id", "partition_id", "cid"):
            if key in node_config:
                return int(node_config[key])
    return int(str(cid_or_context))


def make_client_fn(
    *,
    data_dir: Path,
    partition: str,
    clients: Sequence[str],
    model_config: ModelConfig,
    train_config: TrainConfig,
    generation_config: GenerationConfig,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    max_train_examples: Optional[int],
    max_eval_examples: Optional[int],
    seed: int,
    generate_during_rounds: bool,
) -> Callable[[str], fl.client.Client]:
    def client_fn(cid: str) -> fl.client.Client:
        client_idx = _client_index(cid)
        setup_hpc_runtime(seed + client_idx)
        client_name = clients[client_idx]
        model, tokenizer = setup_lora_model(model_config)
        loaders = build_client_dataloaders(
            data_dir=data_dir,
            partition=partition,
            client_name=client_name,
            tokenizer=tokenizer,
            train_config=train_config,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            max_train_examples=max_train_examples,
            max_eval_examples=max_eval_examples,
            seed=seed,
        )
        return MedicalFLClient(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=loaders["train"],
            val_loss_dataloader=loaders["val_loss"],
            val_gen_dataloader=loaders["val_gen"],
            client_id=client_name,
            train_config=train_config,
            generation_config=generation_config,
            generate_during_rounds=generate_during_rounds,
        ).to_client()

    return client_fn


def weighted_metric_summary(records: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    weighted_keys = {
        "eval_loss",
        "eval_perplexity",
        "rouge1",
        "rouge2",
        "rougeL",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
    }
    summed_keys = {"loss_eval_seconds", "generation_seconds", "num_loss_tokens"}
    metric_keys = sorted(
        {key for record in records.values() for key in record if key != "num_eval_examples"}
    )
    total_examples = sum(float(record.get("num_eval_examples", 0.0)) for record in records.values())
    summary: Dict[str, float] = {"num_eval_examples": float(total_examples)}
    for key in metric_keys:
        values = [float(record[key]) for record in records.values() if isinstance(record.get(key), (int, float))]
        if not values:
            continue
        if key in summed_keys:
            summary[key] = sum(values)
        elif key in weighted_keys:
            numerator = sum(
                float(record.get(key, 0.0)) * float(record.get("num_eval_examples", 0.0))
                for record in records.values()
            )
            summary[key] = numerator / total_examples if total_examples else 0.0
        else:
            summary[key] = sum(values) / len(values)
    return summary


def evaluate_final_global_model(
    *,
    data_dir: Path,
    partition: str,
    clients: Sequence[str],
    final_parameters: NDArrays,
    model_config: ModelConfig,
    train_config: TrainConfig,
    generation_config: GenerationConfig,
    metric_config: MetricConfig,
    eval_batch_size: int,
    num_workers: int,
    max_eval_examples: Optional[int],
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = setup_lora_model(model_config)
    place_model_on_device(model, device)
    set_lora_parameters(model, final_parameters)
    save_adapter_npz(model, output_dir / "final_lora_adapter.npz")

    per_client: Dict[str, Dict[str, float]] = {}
    for client_name in clients:
        loaders = build_client_dataloaders(
            data_dir=data_dir,
            partition=partition,
            client_name=client_name,
            tokenizer=tokenizer,
            train_config=train_config,
            train_batch_size=1,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            max_eval_examples=max_eval_examples,
            seed=seed,
        )
        loss_metrics = evaluate_cross_entropy_loss(model, loaders["test_loss"], device)
        generation_metrics = evaluate_generation_metrics(
            model=model,
            tokenizer=tokenizer,
            dataloader=loaders["test_gen"],
            generation_config=generation_config,
            device=device,
            metric_config=metric_config,
        )
        per_client[client_name] = {**loss_metrics, **generation_metrics}

    centralized_loaders = build_centralized_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        train_config=train_config,
        train_batch_size=1,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        max_eval_examples=max_eval_examples,
        seed=seed,
    )
    centralized_test = {
        **evaluate_cross_entropy_loss(model, centralized_loaders["test_loss"], device),
        **evaluate_generation_metrics(
            model=model,
            tokenizer=tokenizer,
            dataloader=centralized_loaders["test_gen"],
            generation_config=generation_config,
            device=device,
            metric_config=metric_config,
        ),
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {
        "per_client_test": per_client,
        "weighted_per_client_test": weighted_metric_summary(per_client),
        "centralized_test": centralized_test,
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adapter-only Flower FedAvg for the MIMIC-IV clinical summarization proposal."
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Prepared data directory from data_pipeline.py.")
    parser.add_argument("--partition", choices=["noniid", "iid"], default="noniid")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-cpus-per-client", type=float, default=8.0)
    parser.add_argument("--num-gpus-per-client", type=float, default=1.0)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-final-test", action="store_true")
    parser.add_argument("--eval-generation-every-round", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--bertscore-model-type", type=str, default=None)
    parser.add_argument("--bertscore-batch-size", type=int, default=16)

    parser.add_argument("--model-name", type=str, default=ModelConfig.model_name)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument("--ray-temp-dir", type=Path, default=None)

    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
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
        args.output_dir = Path("results") / f"fl_{args.partition}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_hpc_runtime(args.seed)

    clients = discover_clients(args.data_dir, args.partition)
    model_config = ModelConfig(
        model_name=args.model_name,
        cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir else None,
    )
    train_config = TrainConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.local_epochs,
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

    print(f"[fl] clients ({args.partition}): {clients}")
    print("[fl] building initial adapter payload")
    initial_model, _ = setup_lora_model(model_config)
    initial_arrays = get_lora_parameters(initial_model)
    initial_parameters = ndarrays_to_parameters(initial_arrays)
    initial_adapter_mb = adapter_size_mb_from_arrays(initial_arrays)
    del initial_model, initial_arrays
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    round_logger = RoundLogger(args.output_dir)
    strategy = TrackingFedAvg(
        round_logger=round_logger,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=len(clients),
        min_evaluate_clients=len(clients),
        min_available_clients=len(clients),
        on_fit_config_fn=fit_config_fn(args.local_epochs),
        initial_parameters=initial_parameters,
    )

    start = time.time()
    ray_init_args: Dict[str, Any] = {"ignore_reinit_error": True, "include_dashboard": False}
    if args.ray_address:
        ray_init_args["address"] = args.ray_address
    if args.ray_temp_dir:
        args.ray_temp_dir.mkdir(parents=True, exist_ok=True)
        ray_init_args["_temp_dir"] = str(args.ray_temp_dir)

    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(
            data_dir=args.data_dir,
            partition=args.partition,
            clients=clients,
            model_config=model_config,
            train_config=train_config,
            generation_config=generation_config,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            max_train_examples=args.max_train_examples,
            max_eval_examples=args.max_eval_examples,
            seed=args.seed,
            generate_during_rounds=args.eval_generation_every_round,
        ),
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": args.num_cpus_per_client,
            "num_gpus": args.num_gpus_per_client,
        },
        ray_init_args=ray_init_args,
    )
    total_training_time = time.time() - start
    round_summary = round_logger.write()

    final_test = None
    if not args.skip_final_test:
        if strategy.final_parameters is None:
            raise RuntimeError("No final aggregated parameters were captured.")
        final_test = evaluate_final_global_model(
            data_dir=args.data_dir,
            partition=args.partition,
            clients=clients,
            final_parameters=parameters_to_ndarrays(strategy.final_parameters),
            model_config=model_config,
            train_config=train_config,
            generation_config=generation_config,
            metric_config=metric_config,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            max_eval_examples=args.max_eval_examples,
            seed=args.seed,
            output_dir=args.output_dir,
        )

    results = {
        "experiment": "federated_lora",
        "partition": args.partition,
        "config": vars(args),
        "clients": list(clients),
        "initial_adapter_size_mb": initial_adapter_mb,
        "system_metrics": {
            "total_training_time_seconds": float(total_training_time),
            **round_summary,
        },
        "history": {
            "losses_distributed": _json_safe(getattr(history, "losses_distributed", [])),
            "losses_centralized": _json_safe(getattr(history, "losses_centralized", [])),
            "metrics_distributed_fit": _json_safe(getattr(history, "metrics_distributed_fit", {})),
            "metrics_distributed": _json_safe(getattr(history, "metrics_distributed", {})),
            "metrics_centralized": _json_safe(getattr(history, "metrics_centralized", {})),
        },
        "final_test": final_test,
    }
    with (args.output_dir / "federated_results.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(results), handle, indent=2, ensure_ascii=False)

    print("=" * 72)
    print(f"Federated {args.partition} run finished")
    print(f"Results: {args.output_dir / 'federated_results.json'}")
    print(f"Round metrics: {args.output_dir / 'round_metrics.csv'}")
    print(f"Training wall-clock seconds: {total_training_time:.2f}")
    if final_test is not None:
        print("Final centralized test:")
        print(json.dumps(_json_safe(final_test["centralized_test"]), indent=2, ensure_ascii=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
