#!/usr/bin/env python3
"""
Improved FinFact image downloader.
- Uses multiple Google Drive download methods
- Adds retries with exponential backoff
- Supports resuming from previous failed downloads
- Converts SVG to PNG

Usage:
    python datasets/finfact/process_dataset_mt3.py \
        --input /path/to/finfact_hf.json \
        --output_dir /path/to/output/
"""

import os
import json
import re
import time
import argparse
import requests
from urllib.parse import urlparse, parse_qs
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    import cairosvg
    HAS_CAIROSVG = True
except ImportError:
    print("Warning: cairosvg not installed. SVG files will be skipped. Install with: pip install cairosvg")
    HAS_CAIROSVG = False

# --- CONFIG ---
MAX_WORKERS  = 16  # reduced to avoid rate limiting
MAX_RETRIES  = 3
RETRY_DELAY  = 2  # seconds


def extract_gdrive_id(url: str) -> str | None:
    """Extract Google Drive file ID from various URL formats."""
    parsed = urlparse(url)
    
    # Only match Google Drive domains
    if not any(domain in parsed.netloc for domain in ['drive.google.com', 'drive.usercontent.google.com']):
        return None
    
    # Format: drive.google.com/uc?export=view&id=XXX
    qs = parse_qs(parsed.query)
    if 'id' in qs and qs['id']:
        return qs['id'][0]
    
    # Format: drive.google.com/file/d/XXX/view
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    
    return None


def download_gdrive_image(file_id: str, dest_folder: str) -> str:
    """
    Download a Google Drive file using multiple methods.
    1. Try drive.usercontent.google.com (works without login)
    2. Try export=download format
    Returns the filename (not full path).
    """
    fname = f"{file_id}.jpg"
    local_fp = os.path.join(dest_folder, fname)
    
    if os.path.exists(local_fp):
        # Verify existing file is valid
        try:
            with Image.open(local_fp) as im:
                im.verify()
            return fname
        except Exception:
            # Invalid file, remove and re-download
            os.remove(local_fp)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    # Method 1: Try drive.usercontent.google.com format (works without login)
    usercontent_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=view&authuser=0"
    try:
        resp = requests.get(usercontent_url, stream=True, timeout=30, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        
        # Check if we got actual image data (not HTML error page)
        content_type = resp.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            raise RuntimeError("Got HTML instead of image")
        
        with open(local_fp, 'wb') as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        
        # Verify it's a valid image
        with Image.open(local_fp) as im:
            im.verify()
        return fname
    except Exception:
        # Clean up failed download
        if os.path.exists(local_fp):
            os.remove(local_fp)
    
    # Method 2: Try the export=download format
    export_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        resp = requests.get(export_url, stream=True, timeout=30, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        
        content_type = resp.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            raise RuntimeError("Got HTML instead of image")
        
        with open(local_fp, 'wb') as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        
        with Image.open(local_fp) as im:
            im.verify()
        return fname
    except Exception:
        if os.path.exists(local_fp):
            os.remove(local_fp)
    
    raise RuntimeError(f"All download methods failed for {file_id}")


def download_regular_image(url: str, dest_folder: str) -> str:
    """
    Download a regular (non-Google Drive) image.
    Returns the filename (not full path).
    """
    parsed = urlparse(url)
    
    # Generate filename from URL
    fname = os.path.basename(parsed.path) or 'image'
    if not os.path.splitext(fname)[1]:
        fname += '.jpg'
    
    # Make filename safe
    fname = re.sub(r'[^\w\-_\.]', '_', fname)[:100]
    
    local_fp = os.path.join(dest_folder, fname)
    
    # Check file extension for special handling
    ext = os.path.splitext(fname)[1].lower()
    is_vector = ext in ['.svg', '.svgz']
    
    # For SVG, we'll convert to PNG - check if PNG version exists
    if is_vector:
        png_fname = os.path.splitext(fname)[0] + '.png'
        png_fp = os.path.join(dest_folder, png_fname)
        if os.path.exists(png_fp):
            try:
                with Image.open(png_fp) as im:
                    im.verify()
                return png_fname
            except Exception:
                os.remove(png_fp)
    elif os.path.exists(local_fp):
        try:
            with Image.open(local_fp) as im:
                im.verify()
            return fname
        except Exception:
            os.remove(local_fp)
    
    # Build headers - include Referer for sites like newspapers.com
    parsed_url = urlparse(url)
    referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': referer,
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    }
    
    resp = requests.get(url, stream=True, timeout=30, headers=headers)
    resp.raise_for_status()
    
    with open(local_fp, 'wb') as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    
    # Verify and process the downloaded file
    if is_vector:
        # Convert SVG to PNG
        if not HAS_CAIROSVG:
            os.remove(local_fp)
            raise RuntimeError(f"Cannot convert SVG without cairosvg: {url!r}")
        
        if os.path.getsize(local_fp) == 0:
            os.remove(local_fp)
            raise RuntimeError(f"Empty SVG file from {url!r}")
        
        # Basic check that it contains SVG content
        with open(local_fp, 'rb') as f:
            content = f.read(500)
            if b'<svg' not in content.lower() and b'<!doctype svg' not in content.lower():
                os.remove(local_fp)
                raise RuntimeError(f"Downloaded file is not a valid SVG: {url!r}")
        
        # Convert SVG to PNG
        png_fname = os.path.splitext(fname)[0] + '.png'
        png_fp = os.path.join(dest_folder, png_fname)
        try:
            cairosvg.svg2png(url=local_fp, write_to=png_fp, output_width=1024)
        except Exception as e:
            if os.path.exists(png_fp):
                os.remove(png_fp)
            os.remove(local_fp)
            raise RuntimeError(f"Failed to convert SVG to PNG: {url!r}: {e}")
        
        # Delete original SVG, keep PNG
        os.remove(local_fp)
        
        # Verify the PNG
        try:
            with Image.open(png_fp) as im:
                im.verify()
        except Exception as ve:
            os.remove(png_fp)
            raise RuntimeError(f"Invalid PNG from SVG conversion {url!r}: {ve}")
        
        return png_fname
    else:
        # For raster images, use PIL verification
        try:
            with Image.open(local_fp) as im:
                im.verify()
        except Exception as ve:
            os.remove(local_fp)
            raise RuntimeError(f"Invalid image from {url!r}: {ve}")
    
    return fname


def download_image_with_retry(url: str, dest_folder: str) -> str:
    """
    Download an image with retry logic.
    Detects Google Drive URLs and uses gdown for those.
    """
    # Skip invalid URLs
    if not url.startswith(('http://', 'https://')):
        raise RuntimeError(f"Invalid URL (no scheme): {url}")
    
    # Check if it's a Google Drive URL
    gdrive_id = extract_gdrive_id(url)
    
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if gdrive_id:
                return download_gdrive_image(gdrive_id, dest_folder)
            else:
                return download_regular_image(url, dest_folder)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # exponential backoff
    
    raise last_error


def assemble_evidence(evid_list: list) -> str:
    parts = []
    for i, ev in enumerate(evid_list, start=1):
        sent = ev.get('sentence', '').strip()
        if sent:
            parts.append(f"Evidence {i}: {sent}")
    return "\n" + "\n".join(parts) if parts else ""


def main():
    parser = argparse.ArgumentParser(description='Download images for FinFact dataset')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSON file (finfact_hf.json)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed dataset')
    args = parser.parse_args()

    # Set up paths
    images_dir = os.path.join(args.output_dir, 'images')
    output_json = os.path.join(args.output_dir, 'finfact.json')
    error_log = os.path.join(args.output_dir, 'download_errors.log')
    os.makedirs(images_dir, exist_ok=True)

    # Load the input JSON records
    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare a flat list of all (record_idx, url) tasks
    tasks = []
    for idx, item in enumerate(data, start=1):
        for img in item.get('image_data', []):
            src = img.get('image_src')
            if src:
                tasks.append((idx, src))

    print(f"Found {len(tasks)} image URLs to download from {len(data)} records")

    # Containers for results
    images_by_record = {i: [] for i in range(1, len(data) + 1)}
    errors = []

    # Download everything in parallel (but with reduced workers for rate limiting)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(download_image_with_retry, url, images_dir): (idx, url)
            for idx, url in tasks
        }

        for future in tqdm(as_completed(future_to_task),
                          total=len(future_to_task),
                          desc="Downloading images"):
            idx, url = future_to_task[future]
            try:
                fname = future.result()
                abs_fp = os.path.abspath(os.path.join(images_dir, fname))
                images_by_record[idx].append(abs_fp)
            except Exception as e:
                err_msg = f"Record {idx}: {url}: {e}"
                errors.append(err_msg)
                tqdm.write(f"❌ {err_msg}")

    # Assemble output records
    out_records = []
    for idx, item in enumerate(data, start=1):
        rec = {
            "claim_id":      idx,
            "image":         images_by_record[idx],
            "claim":         item.get('claim', '').strip(),
            "evidence":      assemble_evidence(item.get('evidence', [])),
            "justification": item.get('justification', '').strip(),
            "label":         item.get('label', '').strip()
        }
        out_records.append(rec)

    # Write to output JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    # Log any errors
    if errors:
        with open(error_log, 'w', encoding='utf-8') as logf:
            logf.write("\n".join(errors))
        print(f"\n⚠️ Completed with {len(errors)} errors (see {error_log})")
    else:
        print("\n✅ All images downloaded successfully.")

    print(f"✔️ Wrote {len(out_records)} records to {output_json}")


if __name__ == '__main__':
    main()
