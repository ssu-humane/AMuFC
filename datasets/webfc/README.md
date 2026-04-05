# WebFC Dataset Preprocessing

This directory contains preprocessing scripts for the WebFC (Web Fact-Checking) dataset.

## Prerequisites

1. Obtain the WebFC dataset files:
   - `webfc.json` - Main dataset file with claims and evidence
   - `images/` - Directory containing evidence images

2. Place them in your data directory:
```
webfc/
├── webfc.json
└── images/
    ├── p_24_0_0.jpg
    ├── p_25_0_0.jpg
    └── ...
```

## Preprocessing

### Process Image Paths

Convert relative image filenames to absolute paths:

```bash
python datasets/webfc/process_dataset.py \
    --input /path/to/webfc/webfc.json \
    --output /path/to/webfc/webfc_processed.json \
    --image_dir /path/to/webfc/images/
```

**Arguments:**
- `--input`: Path to the raw WebFC JSON file
- `--output`: Path for the processed output JSON
- `--image_dir`: Absolute path to the images directory

## Output Format

The processed JSON file contains entries with absolute image paths:

```json
{
  "claim_id": "p_24",
  "claim": "The claim text...",
  "evidence": "Evidence text...",
  "image": "/path/to/webfc/images/p_24_0_0.jpg",
  "label": "refuted"
}
```

## Quick Start

```bash
# Process the dataset
python datasets/webfc/process_dataset.py \
    --input /path/to/webfc/webfc.json \
    --output /path/to/webfc/webfc_processed.json \
    --image_dir /path/to/webfc/images/
```

## Notes

- The script skips images that already have absolute paths (safe to run multiple times)
- Other scripts in this directory (`summary.py`, `refine_claims_gemini.py`, etc.) are internal utilities used for dataset construction

