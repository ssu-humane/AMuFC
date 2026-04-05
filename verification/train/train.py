import os
import json
import yaml
import argparse
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import random
import numpy as np
from dotenv import load_dotenv

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model with a config YAML file.")
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to the config.yaml file')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Path to the dataset config.yaml file')
    return parser.parse_args()

def freeze_vision_encoder(model):
    """
    Freeze the vision encoder while keeping other components trainable
    """
    # For Qwen2-VL, the vision encoder is typically in the visual component
    # For LLaMA 3.2 Vision, it is in vision_model
    for name, param in model.named_parameters():
        if 'visual' in name.lower() or 'vision' in name.lower():
            param.requires_grad = False
            print(f"Frozen parameter: {name}")
        else:
            # Only set requires_grad=True for floating point tensors
            if param.dtype.is_floating_point:
                param.requires_grad = True
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

def format_data(sample:dict, prompt:str, is_llama:bool=False):
    """
    To format original data
    """
    if sample['image']!="":
        try:
            img = Image.open(sample['image'])
            if img.width > 512 or img.height > 512:
                img.thumbnail((512, 512))
            if img.width < 256 or img.height < 256:
                new_w = max(img.width, 256)
                new_h = max(img.height, 256)
                img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        except Exception as e:
            print(f"Error loading image {sample['image']}: {e}")
            img = None
    else:
        img = None
    ret = []
    
    # user
    if img is not None:
        claim = "\nClaim: " + sample["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + sample["evidence"]

        cont = []
        if is_llama:
            # For LLaMA 3.2 Vision: image comes first
            cont.append({'type': 'image', 'image': img})
            cont.append({'type': 'text' , 'text': prompt.replace("[evidence]", "image evidence and text evidence.")+claim+image_evidence})
            cont.append({'type': 'text', 'text': text_evidence})
        else:
            # For Qwen: text -> image -> text
            cont.append({'type': 'text' , 'text': prompt.replace("[evidence]", "image evidence and text evidence.")+claim+image_evidence})
            cont.append({'type': 'image', 'image': img})
            cont.append({'type': 'text', 'text': text_evidence})

        ret.append({'role': 'user', 'content': cont})
    else:
        claim = "\nClaim: " + sample["claim"]
        text_evidence = "\nText Evidence: " + sample["evidence"]

        cont = []
        cont.append({'type': 'text', 'text': prompt.replace("[evidence]", "text evidence.")+claim+text_evidence})

        ret.append({'role': 'user', 'content': cont})

    # assistant
    ret.append({'role': 'assistant', 'content': "verdict: " + sample['label']})

    return {'messages': ret}

def format_selection_data(sample:dict, prompt:str, is_llama:bool=False):
    """
    To format evidence selection data
    """
    if sample['image'] != "":
        try:
            img = Image.open(sample['image'])
            if img.width > 512 or img.height > 512:
                img.thumbnail((512, 512))
            if img.width < 256 or img.height < 256:
                new_w = max(img.width, 256)
                new_h = max(img.height, 256)
                img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        except Exception as e:
            print(f"Error loading image {sample['image']}: {e}")
            img = None
    else:
        img = None
    ret = []

    # user
    if img is not None:
        claim = "\nClaim: " + sample["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + sample["evidence"]

        cont = []
        if is_llama:
            cont.append({'type': 'image', 'image': img})
            cont.append({'type': 'text' , 'text': prompt+claim+image_evidence})
            cont.append({'type': 'text', 'text': text_evidence})
        else:
            cont.append({'type': 'text' , 'text': prompt+claim+image_evidence})
            cont.append({'type': 'image', 'image': img})
            cont.append({'type': 'text', 'text': text_evidence})

        ret.append({'role': 'user', 'content': cont})
    else:
        print("error")
        exit()

    # assistant
    ret.append({'role': 'assistant', 'content': sample['label']})

    return {'messages': ret}


def format_analyze_data(sample:dict, prompt:str, is_llama:bool=False):
    """
    To format analyze data
    """
    if sample['image'] != "":
        try:
            img = Image.open(sample['image'])
            if img.width > 512 or img.height > 512:
                img.thumbnail((512, 512))
            if img.width < 256 or img.height < 256:
                new_w = max(img.width, 256)
                new_h = max(img.height, 256)
                img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        except Exception as e:
            print(f"Error loading image {sample['image']}: {e}")
            img = None
    else:
        img = None
    ret = []

    # user
    if img is not None:
        claim = "\nClaim: " + sample["claim"]
        image_evidence = "\nImage Evidence: "
        text_evidence = "\nText Evidence: " + sample["evidence"]

        cont = []
        if is_llama:
            cont.append({'type': 'image', 'image': img})
            cont.append({'type': 'text' , 'text': prompt+claim+image_evidence})
            cont.append({'type': 'text', 'text': text_evidence})
        else:
            cont.append({'type': 'text' , 'text': prompt+claim+image_evidence})
            cont.append({'type': 'image', 'image': img})
            cont.append({'type': 'text', 'text': text_evidence})

        ret.append({'role': 'user', 'content': cont})
    else:
        print("error")
        exit()

    # assistant
    ret.append({'role': 'assistant', 'content': sample['gpt_output']})

    return {'messages': ret}

def format_all_data(labelpath:str, prompt:str, is_llama:bool, evidence_selection:bool=False, analyze:bool=False):
    with open(labelpath) as j:
        data = json.load(j)

    if evidence_selection:
        print("evidence_selection is True. now train evidence selection model")
        return [format_selection_data(d, prompt, is_llama=is_llama) for d in data]
    elif analyze:
        print("analyze is True. now train analyze model")
        return [format_analyze_data(d, prompt, is_llama=is_llama) for d in data]
    else:
        print("adaptive_fc is False (or removed). now train original task model")
        return [format_data(d, prompt, is_llama=is_llama) for d in data]

if __name__ == "__main__":
    # Parse the arguments
    args = parse_args()

    # Load configuration from the YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset configuration
    with open(args.dataset, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Determine training mode based on config flags
    training_mode = None
    active_modes = []
    
    if config.get('verifier', False):
        active_modes.append('verifier')
        training_mode = 'verifier'
    
    if config.get('analyzer', False):
        active_modes.append('analyzer')
        training_mode = 'analyzer'
    
    if config.get('selector', False):
        active_modes.append('selector')
        training_mode = 'selector'
    
    # Check that exactly one mode is active
    if len(active_modes) == 0:
        raise ValueError("No training mode specified. Set verifier, analyzer, or selector to true in config.")
    elif len(active_modes) > 1:
        raise ValueError(f"Multiple training modes specified: {active_modes}. Only one can be true.")
    
    print(f"Training Mode: {training_mode}")
    
    # Load appropriate training dataset from dataset config
    datapath_key = f"train_{training_mode}_datapath"
    if datapath_key not in dataset_config:
        raise ValueError(f"Dataset config does not contain '{datapath_key}'")
    
    train_datapath = dataset_config[datapath_key]
    print(f"Loading training data from: {train_datapath}")
    
    # Set seed early in __main__
    set_seed(config.get("seed", config["seed"] if "seed" in config else 42))

    load_dotenv()
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")

    # Use values directly from the config where needed
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["bnb_config"]['load_in_4bit'],
        bnb_4bit_use_double_quant=config["bnb_config"]['bnb_4bit_use_double_quant'],
        bnb_4bit_quant_type=config["bnb_config"]['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=torch.bfloat16 if config["bnb_config"]['bnb_4bit_compute_dtype'] == 'bfloat16' else torch.float32
    )

    # LoRA config
    peft_config = LoraConfig(
        lora_alpha=config["peft_config"]['lora_alpha'],
        lora_dropout=config["peft_config"]['lora_dropout'],
        r=config["peft_config"]['r'],
        bias=config["peft_config"]['bias'],
        target_modules=config["peft_config"]['target_modules'],
        task_type=config["peft_config"]['task_type'],
    )

    # Load model and tokenizer
    device_string = f'cuda:{os.getenv("LOCAL_RANK")}'
    print(f"Loading model: {config['model_id']}")
    
    model = AutoModelForVision2Seq.from_pretrained(
        config["model_id"],
        device_map={'': device_string},
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(config["model_id"])
    tokenizer = processor.tokenizer
    
    model_type = model.config.model_type
    is_llama = (model_type == "mllama")
    print(f"Model Type: {model_type}, Is Llama: {is_llama}")

    # Check if vision encoder should be frozen
    if "vision_encoder" in config and config["vision_encoder"].get("freeze", False):
        print("Freezing vision encoder as specified in config...")
        freeze_vision_encoder(model)

    args = SFTConfig(
        deepspeed=config["deepspeed_conf"],
        output_dir=config["save_path"],                   
        num_train_epochs=config["epochs"],                     
        per_device_train_batch_size=config["batch_size"],          
        gradient_accumulation_steps=config["gradient_accumulation_steps"],          
        gradient_checkpointing=True,            
        optim="adamw_torch_fused",              
        logging_steps=1,                        
        save_strategy="steps",                  
        save_steps=7975,
        save_total_limit=1,
        learning_rate=float(config["learning_rate"]),                     
        bf16=True,                              
        tf32=True,                              
        # max_grad_norm=0.3,                      
        warmup_ratio=config["warmup_ratio"],                      
        weight_decay=0.,
        lr_scheduler_type="cosine",
        max_seq_length=config["max_seq_length"],
        report_to="wandb",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # dataset_text_field=config["dataset_text_field"], 
        dataset_kwargs=config["dataset_kwargs"],
        dataloader_num_workers=config["dataloader_num_workers"],
        run_name=config["run_name"] if "run_name" in config else config["save_path"].split("/")[-1],
        seed=config["seed"] if "seed" in config else 42,
        data_seed=config["seed"] if "seed" in config else 42,
        full_determinism=config["full_determinism"] if "full_determinism" in config else False,
    )
    args.remove_unused_columns=False

    # Create a data collator to encode text and image pairs
    def _concat_messages(messages):
        """Helper to concatenate messages as plain text."""
        return processor.tokenizer.apply_chat_template(messages, tokenize=False)

    def collate_fn(examples):
        def mask_non_assistant_tokens(input_ids, labels, examples, tokenizer, is_llama=False):
            for i, example in enumerate(examples):
                messages = example["messages"]
                
                pointer = 0
                if is_llama and len(input_ids[i]) > 0 and tokenizer.bos_token_id is not None and input_ids[i][0] == tokenizer.bos_token_id:
                    pointer += 1
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    if isinstance(content, list):
                        text_parts = [c["text"] for c in content if c["type"] == "text"]
                        message_text = ''.join(text_parts)
                    else:
                        message_text = content
                    
                    if is_llama:
                        formatted_message = f"<|start_header_id|>{role}<|end_header_id|>\n\n{message_text}<|eot_id|>"
                    else:
                        # Qwen ChatML format
                        formatted_message = f"<|im_start|>{role}\n{message_text}<|im_end|>\n"
                    
                    message_tokens = tokenizer(formatted_message, return_tensors='pt').input_ids[0]
                    
                    # Strip BOS if present and not needed for segment calculation (Qwen/Llama might add it)
                    if tokenizer.bos_token_id is not None and len(message_tokens) > 0 and message_tokens[0] == tokenizer.bos_token_id:
                         message_tokens = message_tokens[1:]

                    span_len = len(message_tokens)
                    
                    # For Qwen, the `chatml_message` reconstruction might differ slightly from `apply_chat_template` internals
                    # specifically regarding whitespace or special tokens.
                    # Qwen logic was taken from existing script, assume it works.
                    
                    if role != "assistant":
                        labels[i, pointer:pointer + span_len] = -100
                    pointer += span_len

        # Collating logic
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        
        if is_llama:
            # Llama 3.2 Vision specific manual image extraction
            image_inputs = []
            for example in examples:
                images = []
                for message in example["messages"]:
                    if isinstance(message["content"], list):
                        for content in message["content"]:
                            if content["type"] == "image":
                                images.append(content["image"])
                image_inputs.append(images[0] if images else None)
        else:
            # Qwen / Auto-process
            image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

        # Tokenize and process
        if all(img is None for img in image_inputs):
            batch = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        else:
            batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True, truncation=True)

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Mask image tokens
        if isinstance(processor, Qwen2VLProcessor) or (not is_llama):
             if hasattr(processor, 'image_token_id'):
                 image_token_id = processor.image_token_id
             elif hasattr(processor.tokenizer, 'image_token_id'):
                 image_token_id = processor.tokenizer.image_token_id
             else:
                 image_tokens = [151652, 151653, 151655]
                 image_token_id = None
            
             if image_token_id is not None:
                 labels[labels == image_token_id] = -100
             elif 'image_tokens' in locals():
                 for tid in image_tokens:
                     labels[labels == tid] = -100
        else:
            # Llama
            if hasattr(processor, 'image_token_id'):
                image_token_id = processor.image_token_id
                labels[labels == image_token_id] = -100
            elif hasattr(processor.tokenizer, 'image_token_id'):
                image_token_id = processor.tokenizer.image_token_id
                labels[labels == image_token_id] = -100

        # Mask non-assistant tokens (Unified)
        mask_non_assistant_tokens(input_ids, labels, examples, processor.tokenizer, is_llama=is_llama)

        batch["labels"] = labels
        return batch

    # Load training dataset using train_datapath from dataset config
    dataset = []

    if training_mode == 'selector':
        print("Training mode: selector. Training evidence selection model")
        dataset.extend(format_all_data(train_datapath, config["prompt"], is_llama, evidence_selection=True))
    
    elif training_mode == 'analyzer':
        print("Training mode: analyzer. Training analyzer model")
        dataset.extend(format_all_data(train_datapath, config["prompt"], is_llama, analyze=True))

    elif training_mode == 'verifier':
        print("Training mode: verifier. Training verifier model")
        dataset.extend(format_all_data(train_datapath, config["prompt"], is_llama))

    print(f"Dataset Size: {len(dataset)}")

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
    )

    # Train the model
    trainer.train()

    # Save model
    trainer.save_model(config["save_path"])

    # Save the config.yaml file in save_path
    with open(os.path.join(config["save_path"], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
