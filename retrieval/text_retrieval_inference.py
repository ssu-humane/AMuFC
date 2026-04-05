import argparse
import json
import pandas as pd
import numpy as np
import torch
import faiss
import yaml
from tqdm import tqdm
from typing import Dict
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import bm25s
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def transform_query(query: str) -> str:
    """For retrieval, add the prompt for query (not for documents)."""
    return f'Represent this sentence for searching relevant passages: {query}'


def pooling(outputs: torch.Tensor, inputs: Dict, strategy: str = 'cls') -> np.ndarray:
    """Pooling strategy for transformers models."""
    if strategy == 'cls':
        outputs = outputs[:, 0]
    elif strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
    else:
        raise NotImplementedError
    return outputs.detach().cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Text retrieval pipeline using YAML config.")
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to YAML config file')
    return parser.parse_args()


def load_texts(text_csv_path):
    df = pd.read_csv(text_csv_path)
    
    # Read from paragraph column (sentence-level format)
    if 'paragraph' not in df.columns:
        raise ValueError(f"CSV must contain 'paragraph' column. Found columns: {df.columns.tolist()}")
    
    sentences = df['paragraph'].fillna("").astype(str).tolist()
    
    return sentences, sentences  # returning full list and same for file_paths


def compute_text_embeddings(model, queries, model_type=None, tokenizer=None, prompt_name=None, batch_size=256, is_query=False):
    """Compute embeddings using either SentenceTransformer or transformers models."""
    if model_type == 'mxbai':
        # Use transformers directly for mxbai
        embeddings = []
        # Apply query transformation only for queries, not for documents
        processed_queries = [transform_query(q) for q in queries] if is_query else queries
        
        for i in tqdm(range(0, len(processed_queries), batch_size)):
            batch = processed_queries[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            
            # Move to GPU if available
            device = next(model.parameters()).device
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            
            with torch.no_grad():
                outputs = model(**inputs).last_hidden_state
            batch_embeddings = pooling(outputs, inputs, 'cls')
            embeddings.extend(batch_embeddings)
        return embeddings
    else:
        # Standard SentenceTransformer
        embeddings = []
        for i in tqdm(range(0, len(queries), batch_size)):
            batch = queries[i:i + batch_size]
            if prompt_name:
                embeddings.extend(model.encode(batch, prompt_name=prompt_name))
            else:
                embeddings.extend(model.encode(batch))
        return embeddings


def find_similar_texts_faiss(claim_embeddings, text_embeddings, file_paths, top_k=5):
    claim_embeddings = np.array(claim_embeddings)
    text_embeddings = np.array(text_embeddings)

    faiss.normalize_L2(text_embeddings)
    index = faiss.IndexFlatIP(text_embeddings.shape[1])
    index.add(text_embeddings)

    results = {}
    for i, claim_emb in enumerate(tqdm(claim_embeddings)):
        faiss.normalize_L2(claim_emb.reshape(1, -1))
        distances, indices = index.search(claim_emb.reshape(1, -1), top_k)
        results[i] = [(file_paths[idx], float(distances[0][j])) for j, idx in enumerate(indices[0])]
    return results


def bm25_retrieval(queries, corpus, file_paths, top_k=5):
    """Fast BM25 retrieval using bm25s library."""
    print(f"Building BM25 index for {len(corpus)} documents...")
    
    # Tokenize corpus
    print("Tokenizing corpus...")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=True)
    
    # Create BM25 model and index the corpus
    print("Creating BM25 index...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)
    
    # Tokenize queries
    print(f"Tokenizing {len(queries)} queries...")
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=True)
    
    # Retrieve top-k for all queries at once
    print(f"Retrieving top-{top_k} documents for each query...")
    results_array, scores_array = retriever.retrieve(query_tokens, k=top_k, show_progress=True)
    
    # Convert to the expected format
    results = {}
    for i in range(len(queries)):
        # Get the retrieved corpus texts and scores for this query
        retrieved_docs = [corpus[idx] for idx in results_array[i]]
        retrieved_scores = scores_array[i]
        
        # Combine into (text, score) tuples
        results[i] = [(doc, float(score)) for doc, score in zip(retrieved_docs, retrieved_scores)]
    
    return results


def tfidf_retrieval(queries, corpus, file_paths, top_k=5):
    """TF-IDF retrieval using sklearn."""
    # Build TF-IDF vectorizer on corpus
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    corpus_vectors = vectorizer.fit_transform(corpus)
    
    results = {}
    for i, query in enumerate(tqdm(queries, desc="TF-IDF retrieval")):
        # Transform query using the same vectorizer
        query_vector = vectorizer.transform([query])
        
        # Compute cosine similarity
        similarities = sklearn_cosine_similarity(query_vector, corpus_vectors).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Store results with scores
        results[i] = [(file_paths[idx], float(similarities[idx])) for idx in top_indices]
    
    return results


if __name__ == "__main__":
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load model name and type
    model_name = cfg["model_name"]
    model_type = cfg.get("model_type", None)
    
    # Load dataset with claims
    with open(cfg["dataset_path"]) as f:
        dataset = json.load(f)

    queries = [item['claim'] for item in dataset]
    print("First query:", queries[0])

    # Load evidences from CSV
    lines, informations = load_texts(cfg["evidence_csv"])
    
    # Get top_k parameter
    top_k = cfg.get("top_k", 5)
    
    # Route to appropriate retrieval method
    if model_name.lower() == "bm25":
        print("Using BM25 retrieval...")
        retrieval_results = bm25_retrieval(queries, lines, informations, top_k=top_k)
    elif model_name.lower() == "tf-idf":
        print("Using TF-IDF retrieval...")
        retrieval_results = tfidf_retrieval(queries, lines, informations, top_k=top_k)
    else:
        # Neural embedding retrieval (existing approach)
        print(f"Using neural embedding retrieval with model: {model_name}")
        
        tokenizer = None
        if model_type == 'mxbai':
            # Use transformers for mxbai
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
        else:
            # Use SentenceTransformer for standard models
            model = SentenceTransformer(model_name)

        # Encode claim text (with query transformation for mxbai)
        claim_embeddings = compute_text_embeddings(
            model, queries, 
            model_type=model_type, 
            tokenizer=tokenizer, 
            is_query=True
        )

        # Encode evidences (without query transformation)
        text_embeddings = compute_text_embeddings(
            model, lines, 
            model_type=model_type, 
            tokenizer=tokenizer, 
            is_query=False
        )

        # Retrieve top-k evidence for each claim
        retrieval_results = find_similar_texts_faiss(claim_embeddings, text_embeddings, informations, top_k=top_k)

    # Fill in evidence into dataset
    for i, matches in retrieval_results.items():
        evidences = [text for text, _ in matches]
        dataset[i]['evidence'] = "\n".join([f"Evidence {j + 1}: {e}" for j, e in enumerate(evidences)])

    print("First enriched sample:", dataset[0])

    # Save final dataset
    with open(cfg["output_path"], "w") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {cfg['output_path']}")
