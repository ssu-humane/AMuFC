import torch
import os
import argparse
import yaml
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Merge base model and PEFT adapter.")
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to the config.yaml file')
    return parser.parse_args()

# Parse the arguments
args = parse_args()

# Load configuration from the YAML file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Use values from the config
base_model_id = config['model_id']
adapter_path = config['save_path']  # The path to the adapter from the fine-tuning process
merged_path = os.path.join(config['save_path'], 'merged')  # Save the merged model here

# Load processor + tokenizer first
processor = AutoProcessor.from_pretrained(base_model_id)
tokenizer = processor.tokenizer

# Load base model
model = AutoModelForVision2Seq.from_pretrained(base_model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.bfloat16)
merged_model = peft_model.merge_and_unload()

# Save the merged model and processor (tokenizer)
merged_model.save_pretrained(merged_path, safe_serialization=True)
processor.save_pretrained(merged_path)

# Print a message confirming that the merged model has been saved
print(f'Merged model and processor have been saved to {merged_path}')
