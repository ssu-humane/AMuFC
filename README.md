# AMuFC

**A**daptive **Mu**ltimodal **F**act-**C**hecking with Visual Evidence Necessity

Official code for *"Is a Picture Worth a Thousand Words? Adaptive Multimodal Fact-Checking with Visual Evidence Necessity"*.

AMuFC is a multimodal fact-checking framework that adaptively uses visual evidence through two collaborating vision-language models: an **Analyzer** that assesses whether visual evidence is necessary for a claim, and a **Verifier** that predicts veracity conditioned on the Analyzer's assessment.

## Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate amufc

# Configure API keys
cp .env.pub .env
# Edit .env and set your API keys:
#   - WANDB_PROJECT, WANDB_API_KEY (for experiment tracking)
#   - OPENAI_API_KEY (for GPT models)
#   - GEMINI_API_KEY (for Gemini models)
#   - HF_TOKEN (for HuggingFace model access)
```

## Project Structure

```
AMUFC/
├── scripts/                    # Shell scripts and configurations
│   ├── cfg/                    # Configuration files
│   │   ├── dataset/            # Dataset path configurations
│   │   ├── train/              # Training configurations
│   │   ├── inference/          # Inference configurations
│   │   └── retrieval/          # Retrieval configurations
│   ├── train.sh                # Training script
│   ├── inference_verifier.sh   # Verifier inference
│   ├── inference_analyzer.sh   # Analyzer inference
│   ├── inference_selector.sh   # Selector inference
│   ├── inference_zeroshot.sh   # Zero-shot baseline
│   ├── text_retrieval.sh       # Text evidence retrieval
│   └── image_retrieval.sh      # Image evidence retrieval
├── verification/               # Main verification modules
│   ├── train/                  # Training code
│   └── inference/              # Inference code
├── evaluation/                 # Evaluation scripts
│   ├── merge_result.py         # Merge text/multimodal results
│   ├── caculate_verification.py        # Calculate metrics
│   └── caculate_verification_oracle.py # Oracle upper bound
├── retrieval/                  # Evidence retrieval modules
└── datasets/                   # Dataset preprocessing
    ├── mocheg/                 # MOCHEG dataset (train + test)
    ├── finfact/                # FinFact dataset (test only)
    └── webfc/                  # WebFC dataset (test only)
```

## Datasets

We use **MOCHEG** for training and evaluate on **MOCHEG**, **FinFact**, and **WebFC**.

| Dataset | Usage | Description |
|---------|-------|-------------|
| [MOCHEG](datasets/mocheg/README.md) | Train + Test | Multimodal claim verification with text and image evidence |
| [FinFact](datasets/finfact/README.md) | Test | Financial fact-checking with mixed text/image claims |
| [WebFC](datasets/webfc/README.md) | Test | Web-based fact-checking dataset |

See each dataset's README for preprocessing instructions.

## Configuration

All configurations are in `scripts/cfg/`. See [scripts/cfg/README.md](scripts/cfg/README.md) for detailed documentation.

### Dataset Configuration (`scripts/cfg/dataset/`)

```yaml
train_verifier_datapath: data.json   # Training data for verifier
train_analyzer_datapath: data.json   # Training data for analyzer
train_selector_datapath: data.json   # Training data for selector
val_gold_datapath: data.json         # Validation data (gold evidence)
test_gold_datapath: data.json        # Test data (gold evidence)
test_ret_datapath: data.json         # Test data (retrieved evidence)
```

### Training Configuration (`scripts/cfg/train/`)

```yaml
model_id: "Qwen/Qwen2-VL-7B-Instruct"  # HuggingFace model
save_path: checkpoints/model_name      # Where to save checkpoint
verifier: true    # Enable verifier training
analyzer: false   # Enable analyzer training
selector: false   # Enable selector training
```

### Inference Configuration (`scripts/cfg/inference/`)

```yaml
verifier: scripts/cfg/train/verifier_qwen.yaml  # Verifier config
analyzer: scripts/cfg/train/analyzer_llama.yaml # Analyzer config (optional)
selector: null                                   # Selector config (optional)
dataset_config:
  dataset_split: test  # train, val, test
  dataset_info: gold   # gold, ret
save_result_path: result
```

## Usage

### 1. Evidence Retrieval

Retrieve text and image evidence for claims:

```bash
# Text retrieval
./scripts/text_retrieval.sh --config scripts/cfg/retrieval/text_retrieval.yaml

# Image retrieval
./scripts/image_retrieval.sh --config scripts/cfg/retrieval/image_retrieval.yaml
```

### 2. Training

Train verifier, analyzer, or selector models:

```bash
# Train verifier
./scripts/train.sh \
  --config scripts/cfg/train/verifier_qwen.yaml \
  --dataset scripts/cfg/dataset/mocheg.yaml

# Train analyzer
./scripts/train.sh \
  --config scripts/cfg/train/analyzer_llama.yaml \
  --dataset scripts/cfg/dataset/mocheg.yaml

# Train selector
./scripts/train.sh \
  --config scripts/cfg/train/selector_llama.yaml \
  --dataset scripts/cfg/dataset/mocheg.yaml
```

### 3. Inference

Run inference with trained models:

```bash
# Verifier inference
./scripts/inference_verifier.sh \
  --config scripts/cfg/inference/verifier/verifier.yaml \
  --dataset scripts/cfg/dataset/mocheg.yaml

# Analyzer inference
./scripts/inference_analyzer.sh \
  --config scripts/cfg/inference/analyzer/analyzer.yaml \
  --dataset scripts/cfg/dataset/mocheg.yaml

# Selector inference
./scripts/inference_selector.sh \
  --config scripts/cfg/inference/selector/selector.yaml \
  --dataset scripts/cfg/dataset/mocheg.yaml
```

### 4. Zero-Shot Baseline

Run zero-shot inference with API models:

```bash
./scripts/inference_zeroshot.sh \
  --config scripts/cfg/inference/zero-shot/zero_shot.yaml \
  --dataset scripts/cfg/dataset/mocheg.yaml
```

## Pipeline Example

Complete pipeline from retrieval to verification:

```bash
# Step 1: Retrieve evidence
./scripts/text_retrieval.sh --config scripts/cfg/retrieval/text_retrieval.yaml
./scripts/image_retrieval.sh --config scripts/cfg/retrieval/image_retrieval.yaml

# Step 2: Train models
./scripts/train.sh --config scripts/cfg/train/verifier_qwen.yaml --dataset scripts/cfg/dataset/mocheg.yaml
./scripts/train.sh --config scripts/cfg/train/analyzer_llama.yaml --dataset scripts/cfg/dataset/mocheg.yaml

# Step 3: Run analyzer (optional - analyzes image necessity)
./scripts/inference_analyzer.sh --config scripts/cfg/inference/analyzer/analyzer.yaml --dataset scripts/cfg/dataset/mocheg.yaml

# Step 4: Run verifier with analyzer results
./scripts/inference_verifier.sh --config scripts/cfg/inference/verifier/verifier.yaml --dataset scripts/cfg/dataset/mocheg.yaml
```

## Evaluation

### Merge Results

Merge text-only and multimodal verification results:

```bash
# Multimodal mode: Replace text with multimodal results
python evaluation/merge_result.py --mode multimodal \
    --text results/text.json \
    --multimodal results/image.json \
    --output results/merged.json

# Analyzer mode: Use analyzer decisions to select text or multimodal
python evaluation/merge_result.py --mode analyzer \
    --text results/text.json \
    --multimodal results/image.json \
    --analyzer results/analyzer.json \
    --output results/merged.json
```

### Calculate Metrics

Evaluate verification performance:

```bash
# Single prediction file
python evaluation/caculate_verification.py \
    --ground_truth data/test.json \
    --prediction results/pred.json

# All predictions in directory
python evaluation/caculate_verification.py \
    --ground_truth data/test.json \
    --prediction_dir results/
```

### Oracle Analysis

Calculate oracle upper bound (best possible selection between text and multimodal):

```bash
python evaluation/caculate_verification_oracle.py \
    --ground_truth data/test.json \
    --text results/text.json \
    --multimodal results/multimodal.json \
    --output metrics.json
```

## Results

Results are saved to: `result/{dataset}/{split}/{info}/{module}/`
