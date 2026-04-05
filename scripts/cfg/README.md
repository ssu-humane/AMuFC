# Configuration Files

This directory contains all configuration files for training, inference, and retrieval.

## Directory Structure

```
scripts/cfg/
├── dataset/          # Dataset path configurations
├── deepspeed/        # DeepSpeed training configurations
├── inference/        # Inference configurations
│   ├── analyzer/     # Analyzer inference
│   ├── selector/     # Selector inference
│   ├── verifier/     # Verifier inference
│   └── zero-shot/    # Zero-shot baseline inference
├── retrieval/        # Evidence retrieval configurations
└── train/            # Model training configurations
```

## Dataset Configuration

Define paths to training, validation, and test datasets.

```yaml
train_verifier_datapath: # path to verifier training data
train_analyzer_datapath: # path to analyzer training data
train_selector_datapath: # path to selector training data

val_gold_datapath: # path to validation data (gold evidence)
val_ret_datapath:  # path to validation data (retrieved evidence)

test_gold_datapath: # path to test data (gold evidence)
test_ret_datapath:  # path to test data (retrieved evidence)
```

## Training Configuration

Model training configs are in `train/`. Each config specifies:

```yaml
prompt: # task-specific prompt
model_id: # HuggingFace model identifier
save_path: # checkpoint save directory

verifier: true/false  # enable verifier task
analyzer: true/false  # enable analyzer task
selector: true/false  # enable selector task

peft_config:          # LoRA configuration
  lora_alpha: 256
  lora_dropout: 0.05
  r: 128
  bias: "none"
  target_modules: ["q_proj", "v_proj"]
  task_type: "CAUSAL_LM"

bnb_config:           # Quantization configuration
  load_in_4bit: true
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"

deepspeed_conf: # path to deepspeed config or null
epochs: 2
batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 2e-5
warmup_ratio: 0.03
max_seq_length: 6144
seed: 42
```

## Inference Configuration

### Verifier/Analyzer/Selector Inference

```yaml
verifier: # path to verifier train config yaml
analyzer: # path to analyzer train config yaml or null
analyze_path: # path to pre-computed analysis results or null
selector: # path to selector train config yaml or null
selector_path: # path to pre-computed selection results or null
dataset_config:
  dataset_split: test  # train, val, test
  dataset_info: gold   # gold, ret
save_result_path: result
```

### Zero-Shot Inference

```yaml
model_path: # model name (e.g., gemini-2.5-pro, gpt-4o)
task_name: # task type (multimodal, text, analysis)
max_workers: 4

dataset_config:
  dataset_split: test
  dataset_info: gold

save_path: # output directory
index: # starting index (optional)
```

## Retrieval Configuration

### Text Retrieval

```yaml
model_name: # model path or method (bm25, tfidf)
dataset_path: # input dataset json
evidence_csv: # evidence corpus csv
top_k: 5
output_path: # output json path
```

### Image Retrieval

```yaml
model_name: # image retrieval model path
image_dir: # directory containing images
dataset_path: # input dataset json
output_path_top1: # top-1 retrieval output
output_path_topk: # top-k retrieval output
top_k: 5
```

## Usage Examples

### Training
```bash
python scripts/train.py --config scripts/cfg/train/verifier_qwen.yaml \
                        --dataset scripts/cfg/dataset/mocheg.yaml
```

### Inference
```bash
python scripts/inference.py --config scripts/cfg/inference/verifier/verifier.yaml
```

### Retrieval
```bash
python scripts/text_retrieval.py --config scripts/cfg/retrieval/text_retrieval.yaml
python scripts/image_retrieval.py --config scripts/cfg/retrieval/image_retrieval.yaml
```

