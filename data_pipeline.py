from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from config import CLIENT_NAMES, CLIENT_SPLIT_TARGETS, SPLITS, DataSplitConfig, TrainConfig


INSTRUCTION = (
    "Summarize the patient's hospital course and provide brief discharge "
    "instructions based on the medical notes provided."
)

DISCHARGE_COLUMNS = (
    "note_id",
    "subject_id",
    "hadm_id",
    "note_type",
    "note_seq",
    "charttime",
    "storetime",
    "text",
)
SERVICE_COLUMNS = ("subject_id", "hadm_id", "transfertime", "curr_service")
METADATA_FIELDS = (
    "note_id",
    "subject_id",
    "hadm_id",
    "note_type",
    "note_seq",
    "charttime",
    "storetime",
)

COURSE_HEADERS = {"brief hospital course", "hospital course"}
INSTRUCTION_HEADERS = {"discharge instructions"}
COURSE_STOP_HEADERS = {
    "medications on admission",
    "discharge medications",
    "discharge disposition",
    "discharge diagnosis",
    "discharge condition",
    "followup instructions",
    "follow-up instructions",
    "follow up instructions",
    "pending test results",
}
INSTRUCTION_STOP_HEADERS = {
    "followup instructions",
    "follow-up instructions",
    "follow up instructions",
    "pending test results",
}
FORBIDDEN_TARGET_HEADERS = {
    "medications on admission",
    "discharge medications",
    "discharge disposition:",
    "discharge diagnosis",
    "discharge condition",
    "followup instructions",
    "follow-up instructions",
    "follow up instructions",
}


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}") from exc
    return rows


def write_jsonl(rows: Sequence[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# MIMIC-IV data preparation
# ---------------------------------------------------------------------------
def _read_mimic_table(path: Path, usecols: Sequence[str]) -> Any:
    import pandas as pd

    allowed = set(usecols)
    return pd.read_csv(path, usecols=lambda column: column in allowed)


def load_discharge_with_services(discharge_path: Path, services_path: Path) -> Any:
    """Load discharge notes and attach the first recorded hospital service."""
    notes = _read_mimic_table(discharge_path, DISCHARGE_COLUMNS)
    services = _read_mimic_table(services_path, SERVICE_COLUMNS)

    required_note_cols = {"subject_id", "hadm_id", "text"}
    required_service_cols = {"hadm_id", "curr_service"}
    if not required_note_cols.issubset(notes.columns):
        missing = sorted(required_note_cols - set(notes.columns))
        raise ValueError(f"{discharge_path} is missing required columns: {missing}")
    if not required_service_cols.issubset(services.columns):
        missing = sorted(required_service_cols - set(services.columns))
        raise ValueError(f"{services_path} is missing required columns: {missing}")

    if "transfertime" in services.columns:
        import pandas as pd

        services["transfertime"] = pd.to_datetime(services["transfertime"], errors="coerce")
        services = services.sort_values(["hadm_id", "transfertime"], na_position="last")
    services = services.dropna(subset=["hadm_id", "curr_service"])
    services = services.drop_duplicates(subset=["hadm_id"], keep="first")

    merged = notes.merge(services[["hadm_id", "curr_service"]], on="hadm_id", how="inner")
    merged = merged.dropna(subset=["subject_id", "hadm_id", "text", "curr_service"])
    for column in ("subject_id", "hadm_id"):
        merged[column] = merged[column].astype("int64")
    return merged


def normalize_note_text(text: Any) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    text = re.sub(r"\n[ \t]+\n", "\n\n", text)
    return text.strip()


def canonicalize_header(line: str) -> str:
    line = re.sub(r"\s+", " ", line.strip())
    return line.rstrip(":").strip().lower()


def find_section_start(lines: Sequence[str], candidate_headers: set[str]) -> Optional[int]:
    for index, line in enumerate(lines):
        if canonicalize_header(line) in candidate_headers:
            return index
    return None


def collect_section_until_stop(
    lines: Sequence[str],
    start_index: int,
    stop_headers: set[str],
) -> str:
    collected: List[str] = []
    for index in range(start_index + 1, len(lines)):
        if canonicalize_header(lines[index]) in stop_headers:
            break
        collected.append(lines[index])
    return "\n".join(collected).strip()


def extract_summary_input(text: Any) -> Optional[str]:
    """Use all content before Brief Hospital Course/Hospital Course as source."""
    lines = normalize_note_text(text).split("\n")
    start_index = find_section_start(lines, COURSE_HEADERS)
    if start_index is None:
        return None

    source = "\n".join(lines[:start_index]).strip()
    if not source:
        return None

    lower_source = source.lower()
    if "brief hospital course" in lower_source or "discharge instructions" in lower_source:
        return None
    return source


def extract_summary_target(text: Any) -> Optional[str]:
    """Extract Hospital Course plus optional Discharge Instructions as target."""
    lines = normalize_note_text(text).split("\n")
    output_sections: List[str] = []

    course_index = find_section_start(lines, COURSE_HEADERS)
    if course_index is not None:
        course = collect_section_until_stop(lines, course_index, COURSE_STOP_HEADERS)
        if course:
            output_sections.append(course)

    instruction_index = find_section_start(lines, INSTRUCTION_HEADERS)
    if instruction_index is not None:
        discharge_instructions = collect_section_until_stop(
            lines,
            instruction_index,
            INSTRUCTION_STOP_HEADERS,
        )
        if discharge_instructions:
            output_sections.append(discharge_instructions)

    if not output_sections:
        return None

    target = "\n\n".join(output_sections).strip()
    word_count = len(target.split())
    if len(target) < 50 or word_count < 30 or len(target) > 4000:
        return None

    lower_target = target.lower()
    if any(header in lower_target for header in FORBIDDEN_TARGET_HEADERS):
        return None
    return target


def map_to_client(service: Any) -> str:
    """Map MIMIC curr_service codes into the four proposal clients."""
    value = re.sub(r"[^A-Z0-9]+", "", str(service or "").upper())

    # Important: CMED/CSURG must come before generic MED/SURG checks.
    if value in {"CMED", "CSURG"} or "CARD" in value or "CATH" in value or "CCU" in value:
        return CLIENT_NAMES[2]
    if "MED" in value and not value.startswith(("C", "N")):
        return CLIENT_NAMES[0]
    if "SURG" in value or value in {"ORTHO", "TRAUM", "TRAUMA"}:
        return CLIENT_NAMES[1]
    return CLIENT_NAMES[3]


def build_summarization_examples(df: Any, instruction: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        source = extract_summary_input(row["text"])
        target = extract_summary_target(row["text"])
        if source is None or target is None:
            continue

        example: Dict[str, Any] = {
            "task": "summarization",
            "instruction": instruction,
            "input": source,
            "output": target,
            "curr_service": str(row["curr_service"]),
            "input_char_len": len(source),
            "output_char_len": len(target),
        }
        for field in METADATA_FIELDS:
            if field in row and not _is_missing(row[field]):
                example[field] = _json_scalar(row[field])
        examples.append(example)
    return examples


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except TypeError:
        return False


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def exact_client_split(
    rows: Sequence[Dict[str, Any]],
    *,
    client_name: str,
    partition: str,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return exactly 3500/500/1000 records for one client."""
    required = sum(CLIENT_SPLIT_TARGETS.values())
    if len(rows) < required:
        raise ValueError(
            f"{client_name} has {len(rows)} valid examples, but exact splitting "
            f"requires at least {required}."
        )

    shuffled = [dict(row) for row in rows]
    random.Random(seed).shuffle(shuffled)
    selected = shuffled[:required]

    output: Dict[str, List[Dict[str, Any]]] = {}
    cursor = 0
    for split_name in SPLITS:
        target = CLIENT_SPLIT_TARGETS[split_name]
        split_rows = selected[cursor : cursor + target]
        cursor += target
        for row in split_rows:
            row["split"] = split_name
            row["client_node"] = client_name
            row["partition"] = partition
        output[split_name] = split_rows
    return output


def partition_noniid_exact(
    examples: Sequence[Dict[str, Any]],
    seed: int,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    grouped = {client: [] for client in CLIENT_NAMES}
    for example in examples:
        grouped[map_to_client(example["curr_service"])].append(example)

    partition: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for client_index, client_name in enumerate(CLIENT_NAMES):
        partition[client_name] = exact_client_split(
            grouped[client_name],
            client_name=client_name,
            partition="noniid",
            seed=seed + client_index,
        )
    return partition


def centralized_splits_from_partition(
    partition: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, List[Dict[str, Any]]]:
    splits: Dict[str, List[Dict[str, Any]]] = {split: [] for split in SPLITS}
    for split_name in SPLITS:
        for client_name in CLIENT_NAMES:
            for row in partition[client_name][split_name]:
                splits[split_name].append(
                    dict(row, client_node="Centralized", partition="centralized")
                )
    return splits


def partition_iid(
    splits: Dict[str, Sequence[Dict[str, Any]]],
    seed: int,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    partition = {client: {split: [] for split in SPLITS} for client in CLIENT_NAMES}
    for split_index, (split_name, rows) in enumerate(splits.items()):
        required = CLIENT_SPLIT_TARGETS[split_name] * len(CLIENT_NAMES)
        if len(rows) != required:
            raise ValueError(
                f"Centralized {split_name} has {len(rows)} rows, expected {required}."
            )
        shuffled = [dict(row) for row in rows]
        random.Random(seed + 1000 + split_index).shuffle(shuffled)

        cursor = 0
        for client in CLIENT_NAMES:
            target = CLIENT_SPLIT_TARGETS[split_name]
            client_rows = shuffled[cursor : cursor + target]
            cursor += target
            for row in client_rows:
                row["split"] = split_name
                row["client_node"] = client
                row["partition"] = "iid"
            partition[client][split_name] = client_rows
    return partition


def write_centralized_splits(
    splits: Dict[str, Sequence[Dict[str, Any]]],
    output_dir: Path,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split_name in SPLITS:
        rows = [
            dict(row, client_node="Centralized", partition="centralized")
            for row in splits[split_name]
        ]
        write_jsonl(rows, output_dir / f"centralized_{split_name}.jsonl")
        counts[split_name] = len(rows)
    return counts


def write_partition(
    partition: Dict[str, Dict[str, List[Dict[str, Any]]]],
    output_dir: Path,
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for client_name in CLIENT_NAMES:
        counts[client_name] = {}
        for split_name in SPLITS:
            rows = partition[client_name][split_name]
            write_jsonl(rows, output_dir / f"{client_name}_{split_name}.jsonl")
            counts[client_name][split_name] = len(rows)
    return counts


def prepare_proposal_data(
    discharge_path: Path,
    services_path: Path,
    output_dir: Path,
    split_config: DataSplitConfig,
    instruction: str = INSTRUCTION,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = load_discharge_with_services(discharge_path, services_path)
    examples = build_summarization_examples(merged, instruction)
    noniid_partition = partition_noniid_exact(examples, split_config.seed)
    centralized_splits = centralized_splits_from_partition(noniid_partition)
    iid_partition = partition_iid(centralized_splits, split_config.seed)

    centralized_counts = write_centralized_splits(centralized_splits, output_dir)
    noniid_counts = write_partition(noniid_partition, output_dir / "noniid")
    iid_counts = write_partition(iid_partition, output_dir / "iid")

    initial_by_client: Dict[str, int] = {client: 0 for client in CLIENT_NAMES}
    for service in merged["curr_service"].tolist():
        initial_by_client[map_to_client(service)] += 1

    valid_by_client: Dict[str, int] = {client: 0 for client in CLIENT_NAMES}
    for example in examples:
        valid_by_client[map_to_client(example["curr_service"])] += 1

    manifest = {
        "source": {
            "discharge_path": str(discharge_path),
            "services_path": str(services_path),
        },
        "output_dir": str(output_dir),
        "clients": CLIENT_NAMES,
        "instruction": instruction,
        "split_config": split_config.__dict__,
        "split_policy": {
            "unit": "record",
            "exact_per_client": CLIENT_SPLIT_TARGETS,
            "patient_level_exclusive": False,
            "note": (
                "Exact 3500/500/1000 records per client is prioritized. "
                "A subject with multiple notes may appear in more than one split."
            ),
        },
        "counts": {
            "merged_discharge_service_rows": int(len(merged)),
            "valid_summarization_examples": int(len(examples)),
            "centralized": centralized_counts,
            "noniid": noniid_counts,
            "iid": iid_counts,
        },
        "client_counts_before_extraction": initial_by_client,
        "client_counts_after_extraction": valid_by_client,
        "file_contract": {
            "centralized": "centralized_<split>.jsonl",
            "federated": "<partition>/<client>_<split>.jsonl",
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def resolve_mimic_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.mimic_root is not None:
        discharge_path = args.discharge_path or args.mimic_root / "mimiciv_note" / "discharge.csv.gz"
        services_path = args.services_path or args.mimic_root / "mimiciv" / "hosp" / "services.csv.gz"
    else:
        discharge_path = args.discharge_path
        services_path = args.services_path

    if discharge_path is None or services_path is None:
        raise ValueError(
            "Provide either --mimic-root, or both --discharge-path and --services-path."
        )
    if not discharge_path.exists():
        raise FileNotFoundError(f"Missing discharge file: {discharge_path}")
    if not services_path.exists():
        raise FileNotFoundError(f"Missing services file: {services_path}")
    return discharge_path, services_path


# ---------------------------------------------------------------------------
# Prompting, datasets, and dataloaders
# ---------------------------------------------------------------------------
def filter_summarization_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("task") != "summarization":
            continue
        if row.get("instruction") and row.get("input") and row.get("output"):
            kept.append(row)
    return kept


def sample_rows(
    rows: Sequence[Dict[str, Any]],
    max_examples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    rows = list(rows)
    if max_examples is None or max_examples >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    keep = sorted(indices[:max_examples])
    return [rows[index] for index in keep]


def build_chat_messages(instruction: str, input_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful clinical summarization assistant."},
        {"role": "user", "content": f"{instruction.strip()}\n\n{input_text.strip()}"},
    ]


def render_prompt_text(tokenizer: Any, instruction: str, input_text: str) -> str:
    messages = build_chat_messages(instruction, input_text)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        "System: You are a helpful clinical summarization assistant.\n\n"
        f"User: {instruction.strip()}\n\n{input_text.strip()}\n\n"
        "Assistant:"
    )


def _tokenize_prompt(tokenizer: Any, prompt_text: str, train_config: TrainConfig) -> List[int]:
    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=train_config.truncation,
        max_length=train_config.max_source_length,
        padding=False,
        return_attention_mask=False,
    )
    return list(encoded["input_ids"])


def _tokenize_target(tokenizer: Any, target_text: str, train_config: TrainConfig) -> List[int]:
    eot_id = _assistant_end_token_id(tokenizer)
    max_target_tokens = train_config.max_target_length - 1 if eot_id is not None else train_config.max_target_length
    encoded = tokenizer(
        target_text.strip(),
        add_special_tokens=False,
        truncation=train_config.truncation,
        max_length=max(1, max_target_tokens),
        padding=False,
        return_attention_mask=False,
    )
    ids = list(encoded["input_ids"])
    if eot_id is not None:
        ids.append(eot_id)
    return ids


def _assistant_end_token_id(tokenizer: Any) -> Optional[int]:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
            return token_id
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    return eos_token_id if isinstance(eos_token_id, int) and eos_token_id >= 0 else None


class SummarizationTrainDataset:
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer: Any,
        train_config: TrainConfig,
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.train_config = train_config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        row = self.rows[index]
        prompt_text = render_prompt_text(self.tokenizer, str(row["instruction"]), str(row["input"]))
        prompt_ids = _tokenize_prompt(self.tokenizer, prompt_text, self.train_config)
        target_ids = _tokenize_target(self.tokenizer, str(row["output"]), self.train_config)
        input_ids = prompt_ids + target_ids
        labels = ([-100] * len(prompt_ids)) + target_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SummarizationEvalDataset:
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer: Any,
        train_config: TrainConfig,
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.train_config = train_config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        row = self.rows[index]
        prompt_text = render_prompt_text(self.tokenizer, str(row["instruction"]), str(row["input"]))
        prompt_ids = _tokenize_prompt(self.tokenizer, prompt_text, self.train_config)
        example: Dict[str, Any] = {
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(prompt_ids), dtype=torch.long),
            "reference": str(row["output"]).strip(),
        }
        for key in ("subject_id", "hadm_id", "note_id", "client_node", "curr_service", "split"):
            if key in row:
                example[key] = row.get(key)
        return example


def _left_pad_tensor_list(tensors: Sequence[Any], pad_value: int) -> Any:
    import torch

    max_len = max(tensor.size(0) for tensor in tensors)
    output = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for index, tensor in enumerate(tensors):
        output[index, max_len - tensor.size(0) :] = tensor
    return output


class TrainCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        from torch.nn.utils.rnn import pad_sequence

        return {
            "input_ids": pad_sequence(
                [item["input_ids"] for item in batch],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [item["attention_mask"] for item in batch],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [item["labels"] for item in batch],
                batch_first=True,
                padding_value=-100,
            ),
        }


class EvalCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        collated: Dict[str, Any] = {
            "input_ids": _left_pad_tensor_list(
                [item["input_ids"] for item in batch],
                self.tokenizer.pad_token_id,
            ),
            "attention_mask": _left_pad_tensor_list(
                [item["attention_mask"] for item in batch],
                0,
            ),
            "reference": [item["reference"] for item in batch],
        }
        for key in ("subject_id", "hadm_id", "note_id", "client_node", "curr_service", "split"):
            if key in batch[0]:
                collated[key] = [item.get(key) for item in batch]
        return collated


def _loader_worker_kwargs(num_workers: int) -> Dict[str, Any]:
    if num_workers <= 0:
        return {}
    return {"persistent_workers": True, "prefetch_factor": 2}


def build_train_dataloader(
    jsonl_path: str | Path,
    tokenizer: Any,
    train_config: TrainConfig,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> Any:
    from torch.utils.data import DataLoader

    rows = sample_rows(filter_summarization_rows(read_jsonl(jsonl_path)), max_examples, seed)
    return DataLoader(
        SummarizationTrainDataset(rows, tokenizer, train_config),
        batch_size=batch_size,
        shuffle=shuffle and len(rows) > 0,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=TrainCollator(tokenizer),
        **_loader_worker_kwargs(num_workers),
    )


def build_eval_dataloader(
    jsonl_path: str | Path,
    tokenizer: Any,
    train_config: TrainConfig,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> Any:
    from torch.utils.data import DataLoader

    rows = sample_rows(filter_summarization_rows(read_jsonl(jsonl_path)), max_examples, seed)
    return DataLoader(
        SummarizationEvalDataset(rows, tokenizer, train_config),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=EvalCollator(tokenizer),
        **_loader_worker_kwargs(num_workers),
    )


def partition_dir(data_dir: str | Path, partition: str) -> Path:
    return Path(data_dir) / partition


def centralized_file(data_dir: str | Path, split: str) -> Path:
    path = Path(data_dir) / f"centralized_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing centralized {split} split: {path}")
    return path


def discover_clients(data_dir: str | Path, partition: str = "noniid") -> List[str]:
    manifest = Path(data_dir) / "manifest.json"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        clients = payload.get("clients")
        if isinstance(clients, list) and clients:
            return [str(client) for client in clients]
    return list(CLIENT_NAMES)


def build_client_dataloaders(
    data_dir: str | Path,
    partition: str,
    client_name: str,
    tokenizer: Any,
    train_config: TrainConfig,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_train_examples: Optional[int] = None,
    max_eval_examples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    base_dir = partition_dir(data_dir, partition)
    paths = {split: base_dir / f"{client_name}_{split}.jsonl" for split in SPLITS}
    for split_name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {partition}/{client_name} {split_name}: {path}")

    return {
        "train": build_train_dataloader(
            paths["train"],
            tokenizer,
            train_config,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            max_examples=max_train_examples,
            seed=seed,
        ),
        "val_loss": build_train_dataloader(
            paths["val"],
            tokenizer,
            train_config,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            max_examples=max_eval_examples,
            seed=seed,
        ),
        "val_gen": build_eval_dataloader(
            paths["val"],
            tokenizer,
            train_config,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            max_examples=max_eval_examples,
            seed=seed,
        ),
        "test_loss": build_train_dataloader(
            paths["test"],
            tokenizer,
            train_config,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            max_examples=max_eval_examples,
            seed=seed,
        ),
        "test_gen": build_eval_dataloader(
            paths["test"],
            tokenizer,
            train_config,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            max_examples=max_eval_examples,
            seed=seed,
        ),
    }


def build_centralized_dataloaders(
    data_dir: str | Path,
    tokenizer: Any,
    train_config: TrainConfig,
    *,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    max_train_examples: Optional[int] = None,
    max_eval_examples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    return {
        "train": build_train_dataloader(
            centralized_file(data_dir, "train"),
            tokenizer,
            train_config,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            max_examples=max_train_examples,
            seed=seed,
        ),
        "val_loss": build_train_dataloader(
            centralized_file(data_dir, "val"),
            tokenizer,
            train_config,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            max_examples=max_eval_examples,
            seed=seed,
        ),
        "val_gen": build_eval_dataloader(
            centralized_file(data_dir, "val"),
            tokenizer,
            train_config,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            max_examples=max_eval_examples,
            seed=seed,
        ),
        "test_loss": build_train_dataloader(
            centralized_file(data_dir, "test"),
            tokenizer,
            train_config,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            max_examples=max_eval_examples,
            seed=seed,
        ),
        "test_gen": build_eval_dataloader(
            centralized_file(data_dir, "test"),
            tokenizer,
            train_config,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            max_examples=max_eval_examples,
            seed=seed,
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare MIMIC-IV-Note discharge summarization data from "
            "discharge.csv.gz and services.csv.gz."
        )
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=None,
        help="Directory containing mimiciv_note/discharge.csv.gz and mimiciv/hosp/services.csv.gz.",
    )
    parser.add_argument("--discharge-path", type=Path, default=None)
    parser.add_argument("--services-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discharge_path, services_path = resolve_mimic_paths(args)
    split_config = DataSplitConfig(seed=args.seed)
    manifest = prepare_proposal_data(
        discharge_path=discharge_path,
        services_path=services_path,
        output_dir=args.output_dir,
        split_config=split_config,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
