import json
import os
import yaml
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from inference_zeroshot_model import (
    openai_model, gemini_model, gemini_model_nonthink, gemini_model_think,
    claude_model, qwenvl2_model, qwen2_5_vl_model, 
    llama3_vision_model, internvl3_model
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-shot inference using config YAML files.")
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to the config.yaml file')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Path to the dataset config.yaml file')
    return parser.parse_args()


def load_yaml_with_base(file_path):
    """Load YAML file with support for _base inheritance."""
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


def initialize_model(model_path):
    """Initialize the appropriate model based on model_path."""
    # Gemini models - check for thinking mode
    if "gemini" in model_path.lower():
        if "2.0-flash" in model_path or "2.5-flash" in model_path:
            # Flash models use non-thinking mode
            return gemini_model_nonthink(model_path)
        elif "2.0-pro" in model_path or "2.5-pro" in model_path:
            # Pro models use thinking mode
            return gemini_model_think(model_path)
        else:
            # Default to base gemini_model (OpenAI-compatible API)
            return gemini_model(model_path)
    
    # Other models
    elif "Llama-3.2" in model_path:
        return llama3_vision_model(model_path)
    elif "Qwen2-VL" in model_path:
        return qwenvl2_model(model_path)
    elif "Qwen2.5-VL" in model_path or "Qwen2_5-VL" in model_path:
        return qwen2_5_vl_model(model_path)
    elif "gpt-4o" in model_path:
        return openai_model(model_path)
    elif "claude" in model_path:
        return claude_model(model_path)
    elif "InternVL3" in model_path:
        return internvl3_model(model_path)
    else:
        raise ValueError(f"Model {model_path} not supported")


def run_task(sample, model, task_name):
    """
    Worker function to process a single sample.
    Returns a dictionary with the result.
    """
    # For multimodal and other image-based tasks, skip if no image
    if task_name in ["multimodal", "aftercheck", "analyze"] and not sample.get("image"):
        return None
    
    # Call the appropriate model method based on task_name
    if task_name == "text":
        # Ensure image is empty for text task
        sample_copy = sample.copy()
        sample_copy['image'] = ""
        output_text, label = model.generate_text_factcheck(sample_copy)
    elif task_name == "multimodal":
        output_text, label = model.generate_multimodal_factcheck(sample)
    elif task_name == "beforecheck":
        output_text, label = model.generate_beforecheck(sample)
    elif task_name == "aftercheck":
        output_text, label = model.generate_aftercheck(sample)
    elif task_name == "analyze":
        output_text, label = model.generate_analyze(sample)
    else:
        raise ValueError(f"Task {task_name} not supported")
    
    return {
        "claim_id": sample["claim_id"],
        "image": sample.get("image", ""),
        "claim": sample["claim"],
        "evidence": sample["evidence"],
        "gpt_output": output_text,
        "label": label,
    }


def main():
    args = parse_args()
    
    # Load configurations
    config = load_yaml_with_base(args.config)
    dataset_config = load_yaml_with_base(args.dataset)
    
    # Extract parameters from config
    model_path = config['model_path']
    task_name = config['task_name']
    max_workers = config.get('max_workers', 4)
    index = config.get('index', 1)  # Index for the output filename
    save_path = config['save_path']
    
    # Get dataset info from config
    dataset_split = config['dataset_config']['dataset_split']  # train, val, test
    dataset_info = config['dataset_config']['dataset_info']     # gold, ret
    
    # Build datapath key and get data path
    datapath_key = f"{dataset_split}_{dataset_info}_datapath"
    data_path = dataset_config.get(datapath_key)
    
    if not data_path:
        raise ValueError(f"Dataset config does not contain '{datapath_key}'")
    
    print(f"Dataset split: {dataset_split}, Info: {dataset_info}")
    print(f"Loading data from: {data_path}")
    
    # Initialize model
    print(f"Initializing model: {model_path}")
    model = initialize_model(model_path)
    
    # Print the prompt being used
    print(f"Task: {task_name}")
    print(f"Prompt:\n{model.return_prompt(task_name)}\n")
    
    # Load data
    with open(data_path, 'r') as f:
        data_files = json.load(f)
    
    # Extract model name from model_path
    model_name = model_path.split("/")[-1]
    
    # Create save directory
    save_dir = os.path.join(save_path, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Save directory: {save_dir}")
    print(f"Running with {max_workers} workers, output index: {index}")
    
    # Filter samples based on task requirements
    if task_name in ["multimodal", "aftercheck", "analyze"]:
        # Only process samples with images
        todo = [d for d in data_files if d.get("image")]
        print(f"Processing {len(todo)}/{len(data_files)} samples (only those with images)")
    else:
        # Process all samples
        todo = data_files
        print(f"Processing {len(todo)} samples")
    
    # Initialize results array
    results = [None] * len(todo)
    
    # Run multiprocessing inference
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_task, sample, model, task_name): idx
                  for idx, sample in enumerate(todo)}
        
        for fut in tqdm(as_completed(futures), total=len(futures),
                      desc=f"{task_name}"):
            idx = futures[fut]
            try:
                result = fut.result()
                if result is not None:
                    results[idx] = result
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                # Keep as None or create error entry
                results[idx] = None
    
    # Remove None results
    results = [r for r in results if r is not None]
    
    # Save results with index in filename
    output_filename = f"{task_name}_{index}.json"
    output_path = os.path.join(save_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Saved {len(results)} results to {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
