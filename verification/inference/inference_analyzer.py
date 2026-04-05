import os
import json
import yaml
import argparse
from dotenv import load_dotenv
from transformers import MllamaForConditionalGeneration, AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Optional, Tuple
import gc
from concurrent.futures import ThreadPoolExecutor
import time
import re

# Load environment variables from .env file (for HF_TOKEN, etc.)
load_dotenv()

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Inference with trained analyzer model (Llama 3.2 Vision or Qwen2-VL) and config file.")
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to the config.yaml file')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Path to the dataset config.yaml file')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='Batch size for inference (default: 16)')
    parser.add_argument('--max_workers', '-w', type=int, default=16, help='Number of worker threads for image loading (default: 16)')
    return parser.parse_args()

def load_yaml_with_base(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    if '_base' in data:
        base_path = os.path.join(os.path.dirname(file_path), data['_base'])
        with open(base_path, 'r') as bf:
            base_data = yaml.safe_load(bf)
        # Inherit base and overwrite with current config
        base_data.update({k: v for k, v in data.items() if k != '_base'})
        return base_data
    else:
        return data

def preprocess_image(image_path: str) -> Optional[Image.Image]:
    """Preprocess a single image with error handling."""
    try:
        if not image_path or image_path == "":
            return None

        img = Image.open(image_path)

        # Standardize image size logic as per previous scripts
        if img.width > 512 or img.height > 512:
            img.thumbnail((512, 512))

        if img.width < 256 or img.height < 256:
            new_w = max(img.width, 256)
            new_h = max(img.height, 256)
            img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def load_images_batch(data_batch: List[Dict], max_workers: int = 4) -> List[Optional[Image.Image]]:
    """Load and preprocess images in parallel."""
    image_paths = [item['image'] for item in data_batch]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(preprocess_image, image_paths))

    return images

def clean_generated_text(text: str) -> str:
    """Clean up generated text by removing special tokens and unwanted patterns."""
    if not text or text.strip() == "":
        return ""

    # Remove common special tokens (merging lists from both scripts)
    special_tokens = [
        '<|im_end|>',
        '<|endoftext|>',
        '<|im_start|>',
        '<|end|>',
        '<|start|>',
        '<pad>',
        '</s>',
        '<s>',
        '<unk>',
        '<|user|>',
        '<|assistant|>',
        '<|system|>',
        '<|eot_id|>',
        '<|start_header_id|>',
        '<|end_header_id|>'
    ]

    cleaned_text = text
    for token in special_tokens:
        cleaned_text = cleaned_text.replace(token, '')

    # Remove repeated special tokens using regex
    cleaned_text = re.sub(r'<\|[^|]*\|>', '', cleaned_text)
    cleaned_text = re.sub(r'<[^>]*>', '', cleaned_text)

    # Remove excessive whitespace and newlines
    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # Strip leading/trailing whitespace
    cleaned_text = cleaned_text.strip()

    return cleaned_text

def prepare_batch_inputs(data_batch: List[Dict], images_batch: List[Optional[Image.Image]], config: Dict, processor, is_llama: bool) -> Tuple[List[str], List[Image.Image], List[int]]:
    """Prepare batch inputs handling both Llama and Qwen formatting."""
    PROMPT = config["prompt"]

    texts = []
    valid_images = []
    valid_indices = []

    for i, (data, img) in enumerate(zip(data_batch, images_batch)):
        if img is not None:
            claim = "\nClaim: " + data["claim"]
            image_evidence = "\nImage Evidence: "
            text_evidence = "\nText Evidence: " + data["evidence"]

            if is_llama:
                # For Llama 3.2 Vision: image comes first
                cont = [
                    {'type': 'image'},
                    {'type': 'text', 'text': PROMPT + claim + image_evidence},
                    {'type': 'text', 'text': text_evidence} # Merging text parts is safer for tokenization usually, but original script split these. Sticking to original logic.
                ]
            else:
                # For Qwen/Others: Text -> Image -> Text usually works best or follows specific prompt structure
                cont = [
                    {'type': 'text', 'text': PROMPT + claim + image_evidence},
                    {'type': 'image', 'image': img},
                    {'type': 'text', 'text': text_evidence}
                ]

            messages = [{"role": "user", "content": cont}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            texts.append(text)
            valid_images.append(img)
            valid_indices.append(i)

    return texts, valid_images, valid_indices

def analyze_generate_batch(data_batch: List[Dict], images_batch: List[Optional[Image.Image]], model, processor, config: Dict, is_llama: bool, new_tokens: int = 512) -> List[Optional[str]]:
    """Generate analysis for a batch of samples."""
    if not data_batch:
        return []

    # Prepare inputs for samples with images
    texts, valid_images, valid_indices = prepare_batch_inputs(data_batch, images_batch, config, processor, is_llama)

    if not texts:
        # No valid images in this batch
        return [None] * len(data_batch)

    # Process batch
    if is_llama:
        # Llama 3.2 Vision requires images as nested list
        nested_images = [[img] for img in valid_images]
        inputs = processor(text=texts, images=nested_images, return_tensors="pt", padding=True).to("cuda")
    else:
        inputs = processor(text=texts, images=valid_images, return_tensors="pt", padding=True).to("cuda")
    
    initial_length = inputs['input_ids'].shape[1]

    # Generate with batch processing
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=new_tokens,
            do_sample=False,
            top_k=1,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )

    outputs = processor.batch_decode(generated_ids[:, initial_length:], skip_special_tokens=True)

    # Clean and map outputs back to original batch order
    results = [None] * len(data_batch)
    for output, idx in zip(outputs, valid_indices):
        out = output.strip()
        # Only store non-empty results
        results[idx] = out if out else None

    return results

if __name__ == "__main__":
    # Parse the arguments
    args = parse_args()
    
    inference_config = load_yaml_with_base(args.config)
    dataset_config = load_yaml_with_base(args.dataset)
    
    # Load analyzer training config
    analyzer_config_path = inference_config['analyzer']
    analyzer_config = load_yaml_with_base(analyzer_config_path)
    
    # Extract dataset info from inference config
    dataset_split = inference_config['dataset_config']['dataset_split']  # train, val, test
    dataset_info = inference_config['dataset_config']['dataset_info']     # gold, ret
    
    # Get dataset name from dataset config path
    dataset_name = os.path.basename(args.dataset).replace('.yaml', '')
    
    # Get config file name from analyzer config path
    config_file_name = os.path.basename(analyzer_config_path).replace('.yaml', '')
    
    # Determine data path based on dataset_split and dataset_info
    datapath_key = f"{dataset_split}_{dataset_info}_datapath"
    test_datapath = dataset_config.get(datapath_key)
    if not test_datapath:
        raise ValueError(f"Dataset config does not contain '{datapath_key}'")
    
    print(f"Dataset: {dataset_name}, Split: {dataset_split}, Info: {dataset_info}")
    print(f"Loading test data from: {test_datapath}")
    
    # Use analyzer config
    config = analyzer_config.copy()

    # Get batch size from args (command line takes priority over config)
    batch_size = args.batch_size
    if "inference_batch_size" in config:
        batch_size = config["inference_batch_size"]

    max_workers = args.max_workers

    print(f"Using batch size: {batch_size}")
    print(f"Using max workers for image loading: {max_workers}")

    # Determine Model Type - use save_path from analyzer config
    if "peft_config" in config:
        save_path = config['save_path']
        # Check if save_path is a HuggingFace repo ID (format: namespace/repo_name)
        # HF repo IDs don't start with '/' or '.', contain exactly one '/', and don't have path separators after
        if '/' in save_path and not save_path.startswith('/') and not save_path.startswith('.') and save_path.count('/') == 1:
            # This is a HuggingFace repo ID, use it directly
            model_id = save_path
        else:
            # This is a local path, append /merged
            model_id = save_path + "/merged"
    else:
        model_id = config['model_id']

    print(f"Loading model from: {model_id}")
    
    # Allow loose loading for different architectures
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    
    model_type = model.config.model_type
    is_llama = (model_type == "mllama")
    
    print(f"Detected model type: {model_type}")
    print(f"Using {'Llama 3.2 Vision' if is_llama else 'Qwen2-VL / Standard'} input formatting")

    processor = AutoProcessor.from_pretrained(model_id)

    # Set left padding for batch processing
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Load test data
    with open(test_datapath, 'r') as j:
        data_files = json.load(j)

    print(f"Loaded {len(data_files)} samples")

    analyze_results = []
    img_index = 0
    empty_generations = 0

    # Process data in batches
    start_time = time.time()

    for i in tqdm(range(0, len(data_files), batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, len(data_files))
        data_batch = data_files[i:batch_end]

        # Load images for this batch in parallel
        images_batch = load_images_batch(data_batch, max_workers)

        # Generate analysis for the batch
        analyses = analyze_generate_batch(data_batch, images_batch, model, processor, config, is_llama)

        # Store results
        for j, (data, img, analysis) in enumerate(zip(data_batch, images_batch, analyses)):
            # Handle empty or None analysis
            if analysis is None or analysis.strip() == "":
                gpt_output = ""
                if img is not None:
                    empty_generations += 1
                    # print(f"Warning: Empty analysis generated for claim_id {data.get('claim_id', i + j)}")
            else:
                gpt_output = analysis

            result = {
                "claim_id": data.get("claim_id", i + j),
                "claim": data["claim"],
                "evidence": data["evidence"],
                "image": data["image"],
                "gpt_output": gpt_output,
                "has_image": img is not None
            }
            analyze_results.append(result)

            if img is not None:
                img_index += 1

        # Clear GPU cache periodically
        if (i // batch_size) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    end_time = time.time()
    processing_time = end_time - start_time

    # Save Results - new structure: result/{dataset_name}/{dataset_split}/{dataset_info}/{config_file_name}
    base_result_path = inference_config.get('save_result_path', 'result')
    result_dir = os.path.join(base_result_path, dataset_name, dataset_split, dataset_info, config_file_name)
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Saving results to: {result_dir}")

    # Save JSON results
    output_path = os.path.join(result_dir, 'analyze_results.json')
    with open(output_path, 'w') as f:
        json.dump(analyze_results, f, indent=2)

    # Save CSV
    df = pd.DataFrame(analyze_results)
    df.to_csv(os.path.join(result_dir, 'analyze_results.csv'), index=False)

    # Performance statistics
    samples_per_second = len(analyze_results) / processing_time if processing_time > 0 else 0
    images_per_second = img_index / processing_time if processing_time > 0 else 0

    print(f"\n=== Analysis Completed ===")
    print(f"Results saved to: {result_dir}")
    print(f"Total samples processed: {len(analyze_results)}")
    print(f"Samples with images: {img_index}")
    print(f"Empty generations: {empty_generations}")
    print(f"Success rate: {((img_index - empty_generations) / img_index * 100):.1f}%" if img_index > 0 else "N/A")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Speed: {samples_per_second:.2f} samples/sec, {images_per_second:.2f} images/sec")
    print(f"Batch size used: {batch_size}")
