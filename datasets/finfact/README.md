# FinFact Dataset Preprocessing

This directory contains preprocessing scripts for the FinFact (Financial Fact-Checking) dataset.

## Preprocessed Dataset

**We provide the fully preprocessed dataset, so you can skip all preprocessing steps below.**

Download the preprocessed dataset from: [TODO: Add download link]

## Manual Preprocessing (Optional)

If you want to preprocess the dataset yourself, follow the steps below.

### Prerequisites

1. Download the original FinFact dataset (HuggingFace format)
2. Set up API keys in `.env`:
   - `GEMINI_API_KEY` for claim refinement

### Step 1: Download Images

Download images from the original dataset (handles Google Drive links, SVG conversion, etc.):

```bash
python datasets/finfact/process_dataset_mt3.py \
    --input /path/to/finfact_hf.json \
    --output_dir /path/to/finfact/
```

**Output:**
- `finfact.json` - Dataset with downloaded image paths
- `images/` - Directory containing downloaded images
- `download_errors.log` - Log of any failed downloads

### Step 2: Split Text-Only and Image Claims

Extract text-only claims (no images in original data):

```bash
python datasets/finfact/topn_onlytxt.py \
    --input /path/to/finfact.json \
    --original /path/to/finfact_hf.json \
    --output /path/to/finfact_topn_onlytxt.json
```

Extract image claims (with valid images ≥100x100):

```bash
python datasets/finfact/top1_onlyimg.py \
    --input /path/to/finfact.json \
    --output /path/to/finfact_top1_onlyimg.json
```

### Step 3: Refine Ambiguous Claims

Use Gemini to refine vague or incomplete claims:

```bash
python datasets/finfact/refine_claims_gemini.py \
    --input /path/to/finfact_topn_onlytxt.json \
    --output /path/to/finfact_topn_onlytxt_refined.json

python datasets/finfact/refine_claims_gemini.py \
    --input /path/to/finfact_top1_onlyimg.json \
    --output /path/to/finfact_top1_onlyimg_refined.json
```

### Step 4: Merge Datasets

Merge text-only and image claims into final dataset:

```bash
python datasets/finfact/merge_dataset.py \
    --text /path/to/finfact_topn_onlytxt_refined.json \
    --image /path/to/finfact_top1_onlyimg_refined.json \
    --output /path/to/finfact_final.json
```

### Step 5: Create Knowledge Store for Retrieval

Create sentence-level text corpus for evidence retrieval:

```bash
python datasets/finfact/text_knowledge_store.py \
    --input /path/to/finfact.json \
    --output /path/to/finfact_corpus_sentence_level.csv
```

## Output Format

The final dataset JSON contains:

```json
{
  "claim_id": 1,
  "claim": "The refined claim text...",
  "evidence": "Evidence 1: ... Evidence 2: ...",
  "justification": "Explanation of the verdict...",
  "image": "/path/to/images/image.jpg",
  "label": "supported"
}
```

Labels: `supported`, `refuted`, `NEI`

## Quick Start (Full Pipeline)

```bash
# Step 1: Download images
python datasets/finfact/process_dataset_mt3.py --input finfact_hf.json --output_dir ./finfact/

# Step 2: Split by modality
python datasets/finfact/topn_onlytxt.py --input ./finfact/finfact.json --original finfact_hf.json --output ./finfact/text_only.json
python datasets/finfact/top1_onlyimg.py --input ./finfact/finfact.json --output ./finfact/image_only.json

# Step 3: Refine claims
python datasets/finfact/refine_claims_gemini.py --input ./finfact/text_only.json --output ./finfact/text_only_refined.json
python datasets/finfact/refine_claims_gemini.py --input ./finfact/image_only.json --output ./finfact/image_only_refined.json

# Step 4: Merge
python datasets/finfact/merge_dataset.py --text ./finfact/text_only_refined.json --image ./finfact/image_only_refined.json --output ./finfact/finfact_final.json

# Step 5: Create knowledge store
python datasets/finfact/text_knowledge_store.py --input ./finfact/finfact.json --output ./finfact/corpus.csv
```

