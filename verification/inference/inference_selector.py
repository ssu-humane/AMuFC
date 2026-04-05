import os
import json
import yaml
import argparse
from dotenv import load_dotenv
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import gc
from concurrent.futures import ThreadPoolExecutor
import time

# Load environment variables from .env file (for HF_TOKEN, etc.)
load_dotenv()

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Inference with trained evidence selector model (Llama/Qwen) and config file.")
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
        base_data.update({k: v for k, v in data.items() if k != '_base'})
        return base_data
    else:
        return data

def classify_binary(output):
    if not output: return "No"
    output = output.lower()
    if "no" in output:
        return "No"
    elif "yes" in output:
        return "Yes"
    else:
        return "Yes" # Default to Yes if unclear

def preprocess_image(image_path: str) -> Optional[Image.Image]:
    """Preprocess a single image with error handling."""
    try:
        if not image_path or not isinstance(image_path, str) or image_path == "":
            return None
        img = Image.open(image_path)
        if img.width > 512 or img.height > 512:
            img.thumbnail((512, 512))
        if img.width < 256 or img.height < 256:
            new_w = max(img.width, 256)
            new_h = max(img.height, 256)
            img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        return img
    except Exception as e:
        # print(f"Warning: Could not load image {image_path}: {e}")
        return None

def preprocess_claim_images(data: Dict) -> List[Image.Image]:
    """Load and preprocess claim images."""
    claim_images = []
    if data.get("claim_image") and len(data["claim_image"]) > 0:
        for img_path in data["claim_image"]:
            img = preprocess_image(img_path)
            if img:
                claim_images.append(img)
    return claim_images

def load_images_batch(data_batch: List[Dict], max_workers: int = 4) -> Tuple[List[Optional[Image.Image]], List[List[Image.Image]]]:
    """Load evidence images and claim images in parallel."""
    # This is a bit complex because we have evidence_image and list of claim_images.
    # We can just process evidence images in parallel for now as they are guaranteed 1 per sample (or None).
    # Claim images are usually few, maybe load them sequentially or optimize later.
    
    evidence_image_paths = [item['image'] for item in data_batch]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        evidence_images = list(executor.map(preprocess_image, evidence_image_paths))
        
    # Load claim images (sequentially per sample for simplicity, or parallelize if needed)
    # Since claim_images is a list, map might be tricky.
    batch_claim_images = []
    for data in data_batch:
        batch_claim_images.append(preprocess_claim_images(data))
        
    return evidence_images, batch_claim_images

def prepare_inputs_for_sample(data: Dict, img: Optional[Image.Image], claim_imgs: List[Image.Image], 
                              config: Dict, is_llama: bool) -> Tuple[List[Dict], List[Image.Image]]:
    """Prepare content and flatten image list for a single sample."""
    PROMPT = config["prompt"]
    
    # Build Claim Text
    claim_text = "\nClaim text: " + data["claim"] if claim_imgs else "\nClaim: " + data["claim"]
    image_evidence_text = "\nImage Evidence: (last image)"
    text_evidence = "\nText Evidence: " + data["evidence"]
    
    cont = []
    all_images_flat = []
    
    if img is not None:
        if is_llama:
            # Llama 3.2 Vision: Images first (Claim Images -> Evidence Image) -> Text
            # Add claim images
            for c_img in claim_imgs:
                cont.append({'type': 'image'})
                all_images_flat.append(c_img)
            
            # Add evidence image
            cont.append({'type': 'image'})
            all_images_flat.append(img)
            
            # Add text block
            text_block = PROMPT + claim_text
            if claim_imgs:
                 text_block += "\nClaim Images: "
            text_block += image_evidence_text + text_evidence
            
            cont.append({'type': 'text', 'text': text_block})
            
        else:
            # Qwen/Standard: Text -> Images -> Text
            cont.append({'type': 'text', 'text': PROMPT + claim_text})
            
            if claim_imgs:
                cont.append({'type': 'text', 'text': "\nClaim Images: "})
                for c_img in claim_imgs:
                    cont.append({'type': 'image', 'image': c_img})
                    all_images_flat.append(c_img)
            
            cont.append({'type': 'text', 'text': image_evidence_text})
            cont.append({'type': 'image', 'image': img})
            all_images_flat.append(img)
            
            cont.append({'type': 'text', 'text': text_evidence})
            
    return cont, all_images_flat

def prepare_batch_inputs(data_batch: List[Dict], evidence_images: List[Optional[Image.Image]], 
                         claim_images_batch: List[List[Image.Image]], 
                         config: Dict, processor, is_llama: bool) -> Tuple[List[str], List[Any], List[int]]:
    
    texts = []
    valid_indices = []
    
    # Structure for processor. usually:
    # Qwen: images=[img1, img2, ...] flat list matching <image> tokens order in batch? 
    # Or list of lists?
    # Llama: nested list [[img1, img2], [img3]] per sample.
    
    batch_images_struct = [] # This will be passed to `images=` argument
    flat_images_qwen = [] 

    for i, (data, ev_img, cl_imgs) in enumerate(zip(data_batch, evidence_images, claim_images_batch)):
        if ev_img is None:
            # Skip samples with no evidence image (as per original logic)
            continue
            
        cont, sample_images = prepare_inputs_for_sample(data, ev_img, cl_imgs, config, is_llama)
        
        messages = [{"role": "user", "content": cont}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        texts.append(text)
        valid_indices.append(i)
        
        if is_llama:
            # Llama expects list of lists
            batch_images_struct.append(sample_images)
        else:
            # Qwen typically takes flat list of all images in batch
            flat_images_qwen.extend(sample_images)
            
    if not is_llama:
        batch_images_struct = flat_images_qwen
        
    return texts, batch_images_struct, valid_indices

def run_inference_batch(data_batch: List[Dict], evidence_images: List[Optional[Image.Image]], 
                        claim_images_batch: List[List[Image.Image]],
                        model, processor, config: Dict, is_llama: bool,
                        new_tokens: int = 128) -> List[Optional[str]]:
    
    texts, batch_images_struct, valid_indices = prepare_batch_inputs(
        data_batch, evidence_images, claim_images_batch, config, processor, is_llama
    )
    
    if not texts:
        return [None] * len(data_batch)

    try:
        inputs = processor(
            text=texts,
            images=batch_images_struct,
            return_tensors="pt",
            padding=True
        ).to("cuda")
        
        initial_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=new_tokens,
                top_k=1,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        outputs = processor.batch_decode(generated_ids[:, initial_length:], skip_special_tokens=True)
        
        # Map back to batch
        results = [None] * len(data_batch)
        for out, idx in zip(outputs, valid_indices):
            results[idx] = out
            
        return results

    except Exception as e:
        print(f"Batch inference failed: {e}")
        return [None] * len(data_batch)

if __name__ == "__main__":
    args = parse_args()
    inference_config = load_yaml_with_base(args.config)
    dataset_config = load_yaml_with_base(args.dataset)
    
    batch_size = args.batch_size
    max_workers = args.max_workers

    # Load selector training config
    selector_config_path = inference_config['selector']
    selector_config = load_yaml_with_base(selector_config_path)
    
    # Extract dataset info from inference config
    dataset_split = inference_config['dataset_config']['dataset_split']  # train, val, test
    dataset_info = inference_config['dataset_config']['dataset_info']     # gold, ret
    
    # Get dataset name from dataset config path
    dataset_name = os.path.basename(args.dataset).replace('.yaml', '')
    
    # Get config file name from selector config path
    config_file_name = os.path.basename(selector_config_path).replace('.yaml', '')
    
    # Determine data path based on dataset_split and dataset_info
    datapath_key = f"{dataset_split}_{dataset_info}_datapath"
    test_datapath = dataset_config.get(datapath_key)
    if not test_datapath:
        raise ValueError(f"Dataset config does not contain '{datapath_key}'")
    
    print(f"Dataset: {dataset_name}, Split: {dataset_split}, Info: {dataset_info}")
    print(f"Loading test data from: {test_datapath}")
    
    # Use selector config
    config = selector_config.copy()

    # Model Loading - use save_path from selector config
    if config.get("peft_config"):
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
    
    print(f"Loading model: {model_id}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    model_type = model.config.model_type
    is_llama = (model_type == "mllama")
    print(f"Model Type: {model_type}, Is Llama: {is_llama}")

    # Set padding
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Load Data
    with open(test_datapath, 'r') as j:
        data_files = json.load(j)

    selection_results = []
    img_index = 0
    
    print(f"Starting inference on {len(data_files)} samples with batch size {batch_size}...")
    
    for i in tqdm(range(0, len(data_files), batch_size)):
        batch_end = min(i + batch_size, len(data_files))
        data_batch = data_files[i:batch_end]
        
        # Load Images
        evidence_images, claim_images_batch = load_images_batch(data_batch, max_workers)
        
        # Run Inference
        outputs = run_inference_batch(
            data_batch, evidence_images, claim_images_batch, 
            model, processor, config, is_llama
        )
        
        for j, (data, ev_img, output) in enumerate(zip(data_batch, evidence_images, outputs)):
            idx = i + j
            
            selection_label = classify_binary(output) if output is not None else "No"
            
            result = {
                "claim_id": data.get("claim_id", idx),
                "claim": data["claim"],
                "evidence": data["evidence"],
                "image": data["image"],
                "output": output if output is not None else "",
                "label": selection_label,
                "has_image": ev_img is not None
            }
            selection_results.append(result)
            
            if ev_img is not None:
                img_index += 1

        # Clean up
        if i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Save Results - new structure: result/{dataset_name}/{dataset_split}/{dataset_info}/{config_file_name}
    base_result_path = inference_config.get('save_result_path', 'result')
    result_dir = os.path.join(base_result_path, dataset_name, dataset_split, dataset_info, config_file_name)
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Saving results to: {result_dir}")

    # JSON
    output_path = os.path.join(result_dir, 'evidence_selector_results.json')
    with open(output_path, 'w') as f:
        json.dump(selection_results, f, indent=2)

    # CSV
    df = pd.DataFrame(selection_results)
    df.to_csv(os.path.join(result_dir, 'evidence_selector_results.csv'), index=False)
    
    print(f"Evidence selection completed. Results saved to {result_dir}")
    print(f"Total samples processed: {len(selection_results)}")
    print(f"Samples with images: {img_index}")
    
    yes_count = sum(1 for r in selection_results if r["label"] == "Yes")
    no_count = sum(1 for r in selection_results if r["label"] == "No")
    print(f"Selection results - Yes: {yes_count}, No: {no_count}")
