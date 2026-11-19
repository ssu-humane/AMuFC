#!/usr/bin/env python
"""
Process WebFC Dataset
Converts relative image paths to absolute paths.

Usage:
    python datasets/webfc/process_dataset.py \
        --input /path/to/webfc.json \
        --output /path/to/webfc_processed.json \
        --image_dir /path/to/images/
"""

import json
import os
import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile


HF_REPO = "humane-lab/WebFC"
HF_FILENAME = "webfc-fcfilter-url.jsonl"
VERDICTS = {"supported": "supported", "refuted": "refuted", "nei": "NEI"}


def process_dataset(input_file: str, output_file: str, image_base_path: str):
    """Process dataset by converting relative image paths to absolute paths."""
    print(f"Loading dataset from {input_file}...")
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        return

    print(f"Processing {len(data)} items...")

    # Ensure image_base_path ends with separator
    if not image_base_path.endswith(os.sep):
        image_base_path = image_base_path + os.sep

    processed_count = 0
    for item in data:
        if "image" in item and item["image"]:
            # Skip if already an absolute path
            if not os.path.isabs(item["image"]):
                item["image"] = os.path.join(image_base_path, item["image"])
                processed_count += 1

    print(f"Updated {processed_count} image paths")
    print(f"Saving processed dataset to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print("Done.")


def prepare_hf_dataset(input_file, output_file, image_dir=None, labels_file=None, image_zip=None):
    """Convert HF JSONL to AMuFC records, optionally joining verifier targets."""
    with Path(input_file).open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    claim_ids = [row["claim_id"] for row in rows]
    if not rows or len(set(claim_ids)) != len(rows):
        raise ValueError("WebFC input must contain records with unique claim IDs")

    labels = None
    if labels_file is not None:
        labels = json.loads(Path(labels_file).read_text(encoding="utf-8"))
        if not isinstance(labels, dict):
            raise ValueError("Labels must be a JSON object mapping claim IDs to verdicts")
        missing = set(claim_ids) - labels.keys()
        if missing:
            raise ValueError(f"Missing verdicts for {len(missing)} claims, e.g. {min(missing)}")

    output_file = Path(output_file)
    if image_zip is not None:
        image_dir = output_file.parent / "webfc-images"
    image_dir = Path(image_dir).resolve() if image_dir else None
    image_filenames = set()
    for row in rows:
        filename = row.get("image_filename")
        if filename:
            if Path(filename).name != filename or filename in {".", ".."}:
                raise ValueError(f"Invalid image filename for {row['claim_id']}")
            image_filenames.add(filename)

        label = labels[row["claim_id"]] if labels is not None else row.get("label")
        if label is not None:
            if not isinstance(label, str) or label.strip().lower() not in VERDICTS:
                raise ValueError(f"Invalid verifier verdict for {row['claim_id']}: {label!r}")
            row["label"] = VERDICTS[label.strip().lower()]

    if image_zip is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(image_zip) as archive:
            members = {filename: archive.getinfo(f"images/{filename}") for filename in image_filenames}
            with TemporaryDirectory(dir=output_file.parent) as temporary:
                staging = Path(temporary)
                for filename, member in members.items():
                    with archive.open(member) as source, (staging / filename).open("wb") as target:
                        shutil.copyfileobj(source, target)
                image_dir.mkdir(parents=True, exist_ok=True)
                for filename in members:
                    (staging / filename).replace(image_dir / filename)

    for row in rows:
        filename = row.get("image_filename")
        image_path = image_dir / filename if image_dir is not None and filename else None
        row["image"] = str(image_path) if image_path is not None and image_path.is_file() else ""

    with Path(output_file).open("w", encoding="utf-8") as output:
        json.dump(rows, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(f"Wrote {len(rows)} records with {sum(bool(row['image']) for row in rows)} local images")
    if any(row.get("label") is None for row in rows):
        print("Some gold verdicts are missing. Use --labels to prepare verifier training data.")


def main():
    parser = argparse.ArgumentParser(description="Prepare WebFC data for AMuFC")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Local legacy JSON or Hugging Face JSONL")
    source.add_argument("--from-hf", action="store_true", help=f"Download from {HF_REPO}")
    parser.add_argument("--output", required=True, help="Path for the processed JSON")
    images = parser.add_mutually_exclusive_group()
    images.add_argument("--image_dir", help="Directory containing locally available images")
    images.add_argument("--image-zip", help="Local WebFC images.zip archive")
    parser.add_argument("--labels", help="JSON object mapping WebFC claim IDs to verifier verdicts")
    args = parser.parse_args()

    if args.from_hf:
        from huggingface_hub import hf_hub_download

        args.input = hf_hub_download(
            repo_id=HF_REPO, filename=HF_FILENAME, repo_type="dataset"
        )
        if args.image_dir is None and args.image_zip is None:
            args.image_zip = hf_hub_download(
                repo_id=HF_REPO, filename="images.zip", repo_type="dataset"
            )
    if args.from_hf or Path(args.input).suffix.lower() == ".jsonl":
        prepare_hf_dataset(args.input, args.output, args.image_dir, args.labels, args.image_zip)
    else:
        if args.labels is not None or args.image_zip is not None:
            parser.error("--labels and --image-zip are supported for Hugging Face JSONL inputs")
        if args.image_dir is None:
            parser.error("--image_dir is required for legacy JSON inputs")
        process_dataset(args.input, args.output, args.image_dir)


if __name__ == "__main__":
    main()
