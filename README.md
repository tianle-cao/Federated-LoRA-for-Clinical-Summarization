# Federated LoRA for Clinical Summarization

This repository implements an adapter-only federated fine-tuning pipeline for
clinical discharge-summary summarization using MIMIC-IV-Note discharge notes
and MIMIC-IV hospital service metadata.

The main research question is whether federated LoRA fine-tuning can approach
centralized LoRA performance while reducing communication cost and keeping
clinical notes decentralized across virtual hospital-service clients.

## Overview

- Task: clinical discharge-summary summarization
- Notes: MIMIC-IV-Note `discharge.csv.gz`
- Service metadata: MIMIC-IV hosp `services.csv.gz`
- Base model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Fine-tuning method: LoRA
- Federated learning framework: Flower FedAvg
- Clients: Medicine, Surgery, Cardiovascular, Other
- Baselines: zero-shot Llama-3.1 and centralized LoRA
- Metrics: validation loss, ROUGE-1/2/L, BERTScore, time, adapter payload size

## Data Access Notice

MIMIC-IV and MIMIC-IV-Note are credentialed datasets. Raw MIMIC files and
derived patient-note JSONL files should not be committed to this repository.

Expected local raw data layout:

```text
data_raw/
  mimiciv_note/
    discharge.csv.gz
  mimiciv/
    hosp/
      services.csv.gz
```

The `.gitignore` file excludes raw data, processed JSONL files, and experiment
outputs.

## Repository Structure

```text
config.py         Shared experiment defaults
data_pipeline.py  MIMIC preprocessing and dataloader builders
modeling.py       Llama, LoRA, training, generation, and metrics
baselines.py      Zero-shot and centralized LoRA experiments
client.py         Federated LoRA experiment with Flower
README.md         Project documentation
```

## Default Experiment Setup

The preprocessing pipeline creates an exact 20,000-record subset.

| Split | Per Client | Total |
| --- | ---: | ---: |
| Train | 3,500 | 14,000 |
| Validation | 500 | 2,000 |
| Test | 1,000 | 4,000 |

The exact per-client targets are defined in `config.py`:

```python
CLIENT_SPLIT_TARGETS = {"train": 3500, "val": 500, "test": 1000}
```

The split is record-level to guarantee exact counts. A subject with multiple
notes may appear in more than one split.

## Environment

Recommended Python version: 3.10.

Install PyTorch first using the CUDA build recommended for your HPC cluster.
Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Llama-3.1-8B-Instruct is a gated Hugging Face model. Accept the model license
on Hugging Face and authenticate before running training:

```bash
huggingface-cli login
```

Alternatively, pass a local model directory with `--model-name`.

## Data Preprocessing

Run preprocessing locally where you have access to the credentialed MIMIC data:

```bash
python data_pipeline.py \
  --mimic-root data_raw \
  --output-dir data_processed
```

You can also pass the two files explicitly:

```bash
python data_pipeline.py \
  --discharge-path /path/to/discharge.csv.gz \
  --services-path /path/to/services.csv.gz \
  --output-dir /path/to/data_processed
```

Expected output:

```text
data_processed/
  manifest.json
  centralized_train.jsonl
  centralized_val.jsonl
  centralized_test.jsonl
  noniid/
    Client_0_Medicine_train.jsonl
    ...
  iid/
    Client_0_Medicine_train.jsonl
    ...
```

## Baselines

Run zero-shot and centralized LoRA baselines:

```bash
python baselines.py --data-dir /path/to/data_processed
```

Defaults:

- zero-shot evaluation with Llama-3.1-8B-Instruct
- centralized LoRA fine-tuning for 3 epochs

If the model is stored locally on HPC:

```bash
python baselines.py \
  --data-dir /path/to/data_processed \
  --model-name /path/to/Meta-Llama-3.1-8B-Instruct
```

Outputs:

```text
results/baselines/
  baseline_results.json
  centralized_lora_adapter.npz
```

## Federated LoRA

Run the primary Non-IID experiment:

```bash
python client.py \
  --data-dir /path/to/data_processed \
  --partition noniid
```

Run the IID control:

```bash
python client.py \
  --data-dir /path/to/data_processed \
  --partition iid
```

Defaults:

- 4 clients
- 3 federated rounds
- 1 local epoch per round
- 1 GPU per client
- adapter-only communication

On a 4-GPU H200 job, the default configuration runs four clients in parallel.

If the model is stored locally on HPC:

```bash
python client.py \
  --data-dir /path/to/data_processed \
  --partition noniid \
  --model-name /path/to/Meta-Llama-3.1-8B-Instruct
```

Outputs:

```text
results/fl_noniid/
  federated_results.json
  round_metrics.json
  round_metrics.csv
  final_lora_adapter.npz
```

## Useful Options

```text
--model-name              Hugging Face model ID or local model path
--hf-cache-dir            Hugging Face cache directory
--output-dir              Custom output directory
--train-batch-size        Training batch size
--eval-batch-size         Evaluation batch size
--num-workers             DataLoader workers
--rounds                  Federated rounds
--local-epochs            Local epochs per federated round
--num-gpus-per-client     GPU allocation per Flower client
--num-cpus-per-client     CPU allocation per Flower client
--eval-generation-every-round
                          Also run generation metrics during each FL round
--ray-address             Existing Ray cluster address
--ray-temp-dir            Ray temporary directory
--skip-bertscore          Skip BERTScore for faster debugging
--skip-final-test         Skip final federated test evaluation
```

Quick baseline smoke test:

```bash
python baselines.py \
  --data-dir /path/to/data_processed \
  --max-train-examples 8 \
  --max-eval-examples 4 \
  --num-epochs 1 \
  --skip-bertscore
```

Quick federated smoke test:

```bash
python client.py \
  --data-dir /path/to/data_processed \
  --partition noniid \
  --rounds 1 \
  --max-train-examples 8 \
  --max-eval-examples 4 \
  --skip-final-test
```

The default batch size is conservative for long clinical notes. On an H200,
try `--train-batch-size 4 --eval-batch-size 4` after a successful smoke test.

## Client Partitioning

The Non-IID partition uses `curr_service` from `services.csv.gz`.

Service mapping:

```text
CMED / CSURG / CARD* / CATH / CCU -> Cardiovascular
SURG / ORTHO / TRAUMA             -> Surgery
MED, excluding C* and N* services  -> Medicine
Everything else                   -> Other
```

## Outputs and Reproducibility

Data preprocessing writes `manifest.json` with source paths, split policy,
client counts, and output file names.

Training scripts save run arguments and metrics in JSON output files.

Federated runs also log per-round client metrics in `round_metrics.csv`.
Communication metrics report the standard one downlink plus one uplink
round-trip adapter payload (`2 * adapter_size_mb`).

The default random seed is `42`.

## Citation

If using MIMIC-IV or MIMIC-IV-Note, cite the official dataset publications and
follow PhysioNet citation and credentialing requirements.
