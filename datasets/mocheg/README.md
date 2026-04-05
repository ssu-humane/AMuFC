# MOCHEG Dataset Preprocessing

This directory contains preprocessing scripts for the MOCHEG (Multimodal Claim Verification) dataset.

## Download

Download the MOCHEG dataset from the official repository:

👉 **https://github.com/PLUM-Lab/Mocheg**

## Prerequisites

1. Clone or download the dataset from the repository above
2. Extract the dataset to a directory (e.g., `/path/to/mocheg`)

The expected directory structure:
```
mocheg/
├── train/
│   ├── Corpus2.csv
│   └── images/
├── val/
│   ├── Corpus2.csv
│   └── images/
└── test/
    ├── Corpus2.csv
    └── images/
```

## Preprocessing Steps

### Step 1: Merge and Clean Evidence

This step merges multiple evidence entries per claim and adds image evidence paths:

```bash
python datasets/mocheg/verification_preprocess.py --data_path /path/to/mocheg
```

**Output:** Creates `Corpus2_for_verification.csv` in each split directory (train/val/test)

### Step 2: Convert to VLM Format

This step converts the CSV data to JSON format suitable for Vision-Language Models:

```bash
python datasets/mocheg/vlm_verification_preprocess.py --data_path /path/to/mocheg
```

**Output:** Creates the following files in each split directory:
- `Ret_Corpus2_for_vlm_single_image.json` - Single image per claim
- `Ret_Corpus2_for_vlm_multi_image.json` - Multiple images per claim

## Output Format

The generated JSON files contain entries with the following structure:

```json
{
  "claim_id": 12345,
  "image": "/path/to/mocheg/test/images/12345-proof-001.jpg",
  "claim": "The claim text...",
  "evidence": "Evidence 1: ... Evidence 2: ...",
  "label": "refuted"
}
```

## Quick Start

```bash
# Full preprocessing pipeline
python datasets/mocheg/verification_preprocess.py --data_path /path/to/mocheg
python datasets/mocheg/vlm_verification_preprocess.py --data_path /path/to/mocheg
```

