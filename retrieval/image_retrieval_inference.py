import argparse
import json
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.nn.functional import cosine_similarity
import faiss
import numpy as np
import yaml
from transformers import CLIPTokenizer, AutoModel, AutoProcessor
from sentence_transformers import SentenceTransformer
from remove_noimg import remove_noimg


def parse_args():
    parser = argparse.ArgumentParser(description="Image retrieval from text claims with config.")
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to YAML config file')
    return parser.parse_args()


def load_images(image_dir):
    file_paths = []
    for file in tqdm(os.listdir(image_dir)):
        if file.lower().endswith(("png", "jpg", "jpeg")):
            file_paths.append(os.path.join(image_dir, file))
    return file_paths


def compute_image_embeddings(model, file_paths, model_type=None, processor=None, batch_size=32):
    """Compute image embeddings using either SentenceTransformer or processor-based model."""
    if model_type == 'siglip':
        # Use processor for SigLIP
        embeddings = []
        device = next(model.parameters()).device
        
        for i in tqdm(range(0, len(file_paths), batch_size)):
            images = [Image.open(path).convert('RGB') for path in file_paths[i:i+batch_size]]
            inputs = processor(images=images, return_tensors="pt", padding="max_length")
            
            # Move to GPU if available
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            
            with torch.no_grad():
                image_outputs = model.get_image_features(**inputs)
            # Normalize embeddings
            image_outputs = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
            embeddings.extend(image_outputs.cpu().numpy())
        return embeddings
    else:
        # Standard SentenceTransformer (CLIP)
        embeddings = []
        for i in tqdm(range(0, len(file_paths), batch_size)):
            images = [Image.open(path) for path in file_paths[i:i+batch_size]]
            embeddings.extend(model.encode(images))
        return embeddings


def truncate_text(tokenizer, model, text):
    """Truncate text for CLIP models."""
    tokens = model.tokenize([text])
    return tokenizer.decode(tokens['input_ids'][0][:75], skip_special_tokens=True)


def compute_text_embeddings(model, queries, model_type=None, tokenizer=None, processor=None, prompt_name=None, batch_size=256):
    """Compute text embeddings using either SentenceTransformer or processor-based model."""
    if model_type == 'siglip':
        # Use processor for SigLIP
        embeddings = []
        device = next(model.parameters()).device
        
        for i in tqdm(range(0, len(queries), batch_size)):
            batch = queries[i:i+batch_size]
            inputs = processor(text=batch, return_tensors="pt", padding="max_length", truncation=True)
            
            # Move to GPU if available
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            
            with torch.no_grad():
                text_outputs = model.get_text_features(**inputs)
            # Normalize embeddings
            text_outputs = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
            embeddings.extend(text_outputs.cpu().numpy())
        return embeddings
    else:
        # Standard SentenceTransformer (CLIP)
        truncated = [truncate_text(tokenizer, model, q) for q in queries]
        embeddings = []
        for i in tqdm(range(0, len(truncated), batch_size)):
            chunk = truncated[i:i+batch_size]
            if prompt_name:
                embeddings.extend(model.encode(chunk, prompt_name=prompt_name))
            else:
                embeddings.extend(model.encode(chunk))
        return embeddings


def find_similar_images_faiss(text_embeddings, image_embeddings, file_paths, top_k=5):
    text_embeddings = np.array(text_embeddings)
    image_embeddings = np.array(image_embeddings)
    faiss.normalize_L2(image_embeddings)
    index = faiss.IndexFlatIP(image_embeddings.shape[1])
    index.add(image_embeddings)

    results = {}
    for i, text_embedding in tqdm(enumerate(text_embeddings)):
        faiss.normalize_L2(text_embedding.reshape(1, -1))
        distances, indices = index.search(text_embedding.reshape(1, -1), top_k)
        results[i] = [(file_paths[idx], float(distances[0][j])) for j, idx in enumerate(indices[0])]
    return results


if __name__ == "__main__":
    args = parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    model_type = cfg.get("model_type", None)
    image_dir = cfg["image_dir"]
    dataset_path = cfg["dataset_path"]
    output_path_top1 = cfg["output_path_top1"]
    output_path_topk = cfg["output_path_topk"]
    top_k = cfg.get("top_k", 5)

    # Load model based on type
    tokenizer = None
    processor = None
    
    if model_type == 'siglip':
        # Use AutoModel and AutoProcessor for SigLIP
        model = AutoModel.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        # Use SentenceTransformer for CLIP
        model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Load dataset and extract queries
    with open(dataset_path) as f:
        dataset = json.load(f)

    queries = [item["claim"] for item in dataset]
    print(f"First query: {queries[0]}")

    # Load and embed images
    file_paths = load_images(image_dir)
    image_embeddings = compute_image_embeddings(
        model, file_paths, 
        model_type=model_type, 
        processor=processor
    )

    # Embed text
    text_embeddings = compute_text_embeddings(
        model, queries, 
        model_type=model_type, 
        tokenizer=tokenizer, 
        processor=processor
    )

    # Find similar images
    retrieval_results = find_similar_images_faiss(text_embeddings, image_embeddings, file_paths, top_k=top_k)

    # Save top-1 result
    for i, matches in retrieval_results.items():
        dataset[i]["image"] = matches[0][0]

    with open(output_path_top1, "w") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved top-1 image result to: {output_path_top1}")

    # Save top-k result
    for i, matches in retrieval_results.items():
        dataset[i]["image"] = [m[0] for m in matches]

    with open(output_path_topk, "w") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved top-{top_k} image results to: {output_path_topk}")