import os
import json
import yaml
import argparse
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import gc
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file (for HF_TOKEN, etc.)
load_dotenv()


# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Inference with trained model and config file.")
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to the config.yaml file')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Path to the dataset config.yaml file')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size for inference (default: 16)')
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

def calc_metric(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    report = classification_report(y_true, y_pred, target_names=['Refuted', 'Supported', 'NEI'], output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=['Refuted', 'Supported', 'NEI'])
    print(report)
    print(report_str)
    return cm, report, report_str

def classify_output(output):
    if not output: return 2 # Treat empty/failure as NEI or error
    output = output.lower()
    if "refuted" in output or "false" in output:
        return 0
    elif "supported" in output or "true" in output:
        return 1
    else:
        return 2

def preprocess_image(image_path: str) -> Optional[Image.Image]:
    """Preprocess a single image with error handling."""
    try:
        if not image_path or image_path == "":
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
        # print(f"Error processing image {image_path}: {e}")
        return None

def load_images_batch(data_batch: List[Dict], max_workers: int = 4) -> List[Optional[Image.Image]]:
    """Load and preprocess images in parallel."""
    image_paths = [item['image'] for item in data_batch]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(preprocess_image, image_paths))
    return images

def prepare_inputs_for_sample(data: Dict, img: Optional[Image.Image], config: Dict, 
                              analyze_data: Optional[Dict], selection_data: Optional[Dict], 
                              is_llama: bool) -> List[Dict]:
    """Prepare chat content for a single sample based on configuration modes."""
    PROMPT = config["prompt"]
    claim = "\nClaim: " + data["claim"]
    text_evidence = "\nText Evidence: " + data["evidence"]
    image_evidence_str = "\nImage Evidence: "
    
    # Determine external data if available
    analyze_output = None
    selection_label = "No" # Default to No if not selection model

    if "analyze_fc" in config and config["analyze_fc"] and analyze_data:
        analyze_output = analyze_data.get(data.get("claim_id"), {}).get("gpt_output")
    
    if "evidence_selection_fc" in config and config["evidence_selection_fc"] and selection_data:
        selection_label = selection_data.get(data.get("claim_id"), {}).get("label", "No")

    # Determine usage logic based on config flags
    use_analyze = ("analyze_fc" in config and config["analyze_fc"])
    use_selection = ("evidence_selection_fc" in config and config["evidence_selection_fc"])

    # Determine if we effectively use the image
    should_use_image = False
    prompt_replacement = "text evidence."

    if img is not None:
        if use_selection and use_analyze:
            # Both enabled: Use image only if selected=Yes. If so, include analysis.
            if selection_label == "Yes":
                should_use_image = True
                prompt_replacement = "image evidence with analysis and text evidence."
                if analyze_output:
                    # Inject analysis into text evidence or separate block?
                    # Original implementation: image_analysis + text_evidence
                    text_evidence = "\nImage Analysis: " + analyze_output + text_evidence
        elif use_selection:
            # Only selection enabled
            if selection_label == "Yes":
                should_use_image = True
                prompt_replacement = "image evidence and text evidence."
        elif use_analyze:
            # Only analysis enabled
            should_use_image = True # Assume we always use image if analysis is on?
            # Original code check: if img is not None and analyze is not None
            if analyze_output:
                prompt_replacement = "image evidence with analysis and text evidence."
                text_evidence = "\nImage Analysis: " + analyze_output + text_evidence
            else:
                should_use_image = True
                prompt_replacement = "image evidence and text evidence."
        else:
            # Standard mode (Original)
            should_use_image = True
            prompt_replacement = "image evidence and text evidence."

    final_prompt = PROMPT.replace("[evidence]", prompt_replacement) + claim

    cont = []
    
    if should_use_image and img is not None:
        # Include Image
        if is_llama:
            # Llama logic: Image -> Text
            cont.append({'type': 'image'})
            cont.append({'type': 'text', 'text': final_prompt + image_evidence_str})
            cont.append({'type': 'text', 'text': text_evidence})
        else:
            # Qwen: Text -> Image -> Text
            cont.append({'type': 'text', 'text': final_prompt + image_evidence_str})
            cont.append({'type': 'image', 'image': img})
            cont.append({'type': 'text', 'text': text_evidence})
    else:
        # Text Only
        cont.append({'type': 'text', 'text': final_prompt + text_evidence})
        
    return cont

def prepare_batch_inputs(data_batch: List[Dict], images_batch: List[Optional[Image.Image]], 
                         config: Dict, processor, is_llama: bool,
                         analyze_data: Optional[Dict], selection_data: Optional[Dict]) -> Tuple[List[str], List[Image.Image], List[int]]:
    """Prepare batch inputs."""
    texts = []
    valid_images = []
    valid_indices = []

    for i, (data, img) in enumerate(zip(data_batch, images_batch)):
        cont = prepare_inputs_for_sample(data, img, config, analyze_data, selection_data, is_llama)
        
        # Check if this sample actually uses an image (based on prepared content)
        has_image = any(item.get('type') == 'image' for item in cont)
        
        messages = [{"role": "user", "content": cont}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        texts.append(text)
        if has_image:
            valid_images.append(img)
        
        valid_indices.append(i)

    return texts, valid_images, valid_indices

def run_inference_batch(data_batch: List[Dict], images_batch: List[Optional[Image.Image]], 
                        model, processor, config: Dict, is_llama: bool,
                        analyze_data: Optional[Dict], selection_data: Optional[Dict],
                        new_tokens: int = 128) -> List[str]:
    
    texts, valid_images, _ = prepare_batch_inputs(data_batch, images_batch, config, processor, is_llama, analyze_data, selection_data)
    
    if not texts:
        return [""] * len(data_batch)

    try:      
        inference_kwargs = {
            "text": texts,
            "return_tensors": "pt",
            "padding": True
        }
        
        if valid_images:
            if is_llama:
                nested_images = [[img] for img in valid_images]
                inference_kwargs["images"] = nested_images
            else:
                # Qwen/Standard
                inference_kwargs["images"] = valid_images
        
        inputs = processor(**inference_kwargs).to("cuda")
        
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
        return outputs

    except Exception as e:
        print(f"Batch inference failed: {e}")
        # Fallback to single? Or return empty strings?
        return [""] * len(data_batch)

if __name__ == "__main__":
    args = parse_args()
    inference_config = load_yaml_with_base(args.config)
    dataset_config = load_yaml_with_base(args.dataset)
    
    batch_size = args.batch_size
    max_workers = args.max_workers

    # Load verifier training config
    verifier_config_path = inference_config['verifier']
    verifier_config = load_yaml_with_base(verifier_config_path)
    
    # Extract dataset info from inference config
    dataset_split = inference_config['dataset_config']['dataset_split']  # train, val, test
    dataset_info = inference_config['dataset_config']['dataset_info']     # gold, ret
    
    # Get dataset name from dataset config path
    dataset_name = os.path.basename(args.dataset).replace('.yaml', '')
    
    # Build result path component based on activated modules
    # Collect module names in order: analyzer, selector, verifier
    result_path_components = []
    
    # Check analyzer
    if inference_config.get('analyzer'):
        analyzer_config_path = inference_config['analyzer']
        analyzer_config_name = os.path.basename(analyzer_config_path).replace('.yaml', '')
        result_path_components.append(analyzer_config_name)
    elif inference_config.get('analyze_path'):  # Old method - use parent directory name
        analyze_path = inference_config['analyze_path']
        parent_dir = os.path.basename(os.path.dirname(analyze_path))
        result_path_components.append(parent_dir)
    
    # Check selector
    if inference_config.get('selector'):
        selector_config_path = inference_config['selector']
        selector_config_name = os.path.basename(selector_config_path).replace('.yaml', '')
        result_path_components.append(selector_config_name)
    elif inference_config.get('selector_path'):  # Old method - use parent directory name
        selector_path = inference_config['selector_path']
        parent_dir = os.path.basename(os.path.dirname(selector_path))
        result_path_components.append(parent_dir)
    
    # Always add verifier
    verifier_config_name = os.path.basename(verifier_config_path).replace('.yaml', '')
    result_path_components.append(verifier_config_name)
    
    # Join with underscores
    config_file_name = '_'.join(result_path_components)
    
    # Determine data path based on dataset_split and dataset_info
    datapath_key = f"{dataset_split}_{dataset_info}_datapath"
    test_datapath = dataset_config.get(datapath_key)
    if not test_datapath:
        raise ValueError(f"Dataset config does not contain '{datapath_key}'")
    
    print(f"Dataset: {dataset_name}, Split: {dataset_split}, Info: {dataset_info}")
    print(f"Loading test data from: {test_datapath}")
    
    # Merge configs - use verifier config as base and add inference-specific settings
    config = verifier_config.copy()
    
    # Override with inference-specific settings if they exist in verifier config
    if 'analyze_fc' in verifier_config:
        config['analyze_fc'] = verifier_config.get('analyze_fc', False)
    if 'analyze_path' in verifier_config:
        config['analyze_path'] = verifier_config.get('analyze_path')
    if 'evidence_selection_fc' in verifier_config:
        config['evidence_selection_fc'] = verifier_config.get('evidence_selection_fc', False)
    if 'evidence_selection_path' in verifier_config:
        config['evidence_selection_path'] = verifier_config.get('evidence_selection_path')

    # Model Loading - use save_path from verifier config
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
        device_map="cuda",
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

    # Load Auxiliary Data - Auto-detect paths or use provided paths
    analyze_data = None
    
    # Check if analyzer is specified in inference config
    if inference_config.get('analyzer'):
        # Auto-detect analyzer result path
        analyzer_config_path = inference_config['analyzer']
        analyzer_config_name = os.path.basename(analyzer_config_path).replace('.yaml', '')
        analyze_result_path = os.path.join(
            inference_config.get('save_result_path', 'result'),
            dataset_name,
            dataset_split,
            dataset_info,
            analyzer_config_name,
            'analyze_results.json'
        )
        
        if os.path.exists(analyze_result_path):
            print(f"Auto-detected Analysis Data: {analyze_result_path}")
            with open(analyze_result_path, 'r') as j:
                analyze_list = json.load(j)
            analyze_data = {item["claim_id"]: item for item in analyze_list}
            config['analyze_fc'] = True
        else:
            print(f"Warning: Auto-detected analyzer results not found at {analyze_result_path}")
    elif inference_config.get('analyze_path'):  # Backward compatibility - direct path
        print(f"Loading Analysis Data from direct path: {inference_config['analyze_path']}")
        with open(inference_config['analyze_path'], 'r') as j:
            analyze_list = json.load(j)
        analyze_data = {item["claim_id"]: item for item in analyze_list}
        config['analyze_fc'] = True

    selection_data = None
    
    # Check if selector is specified in inference config
    if inference_config.get('selector'):
        # Auto-detect selector result path
        selector_config_path = inference_config['selector']
        selector_config_name = os.path.basename(selector_config_path).replace('.yaml', '')
        selector_result_path = os.path.join(
            inference_config.get('save_result_path', 'result'),
            dataset_name,
            dataset_split,
            dataset_info,
            selector_config_name,
            'evidence_selector_results.json'
        )
        
        if os.path.exists(selector_result_path):
            print(f"Auto-detected Selection Data: {selector_result_path}")
            with open(selector_result_path, 'r') as j:
                selection_list = json.load(j)
            selection_data = {item["claim_id"]: item for item in selection_list}
            config['evidence_selection_fc'] = True
        else:
            print(f"Warning: Auto-detected selector results not found at {selector_result_path}")
    elif inference_config.get('selector_path'):  # Backward compatibility - direct path
        print(f"Loading Selection Data from direct path: {inference_config['selector_path']}")
        with open(inference_config['selector_path'], 'r') as j:
            selection_list = json.load(j)
        selection_data = {item["claim_id"]: item for item in selection_list}
        config['evidence_selection_fc'] = True

    # Inference Loop
    true_list = []
    pred_list = []
    
    print(f"Starting inference on {len(data_files)} samples with batch size {batch_size}...")
    
    for i in tqdm(range(0, len(data_files), batch_size)):
        batch_end = min(i + batch_size, len(data_files))
        data_batch = data_files[i:batch_end]
        
        # Load Images
        images_batch = load_images_batch(data_batch, max_workers)
        
        # Run Inference
        outputs = run_inference_batch(
            data_batch, images_batch, model, processor, config, 
            is_llama, analyze_data, selection_data
        )
        
        # Process Results
        for j, output in enumerate(outputs):
            idx = i + j
            data = data_batch[j]
            
            # Classification
            pred_label = classify_output(output)
            pred_list.append(pred_label)
            
            # GT
            gt = data["label"].lower()
            if "refuted" in gt or "false" in gt:
                true_list.append(0)
            elif "supported" in gt or "true" in gt:
                true_list.append(1)
            else:
                true_list.append(2)
        
        # Clean up
        if i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Metrics
    metrics_result = calc_metric(true_list, pred_list)

    # Save Results - new structure: result/{dataset_name}/{dataset_split}/{dataset_info}/{config_file_name}
    base_result_path = inference_config.get('save_result_path', 'result')
    result_dir = os.path.join(base_result_path, dataset_name, dataset_split, dataset_info, config_file_name)
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Saving results to: {result_dir}")
    
    # Save CSV
    df = pd.DataFrame(data_files)
    df['pred_label'] = pred_list
    
    df.to_csv(os.path.join(result_dir, 'inference_results.csv'), index=False)
    
    # Save Summary
    output_path = os.path.join(result_dir, 'inference_summary.txt')
    
    report = metrics_result[1]
    accuracy = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    
    with open(output_path, 'w') as f:
        f.write(f"{accuracy}\n")
        f.write(f"{macro_f1}\n")
        f.write(f"{round(accuracy, 4)}\n")
        f.write(f"{round(macro_f1, 4)}\n")
        
        # Confusion Matrix
        f.write(f"\n{metrics_result[0]}\n")
        
        # Classification Report
        f.write(f"\n{metrics_result[2]}\n")

    print(f"Accuracy: {accuracy}")
    print(f"Macro F1: {macro_f1}")
    print(f"Confusion Matrix:\n{metrics_result[0]}")
    print(f"Classification Report:\n{metrics_result[2]}")
    print(f"Results saved to {result_dir}")