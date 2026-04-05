import argparse
import json
import tqdm
import os, math
from openai import OpenAI
from typing import List
import tiktoken
import copy

MAX_CTX_TOKENS = 128_000
HALF_CTX_TOKENS = MAX_CTX_TOKENS // 2

enc = tiktoken.get_encoding("o200k_base")

PROMPT = """Your task is to read the following document carefully and summarize it into a single, coherent paragraph. Focus on capturing the main ideas and essential details without adding new information or personal opinions.
Document:
{}
"""

def count_tokens(s: str) -> int:
    return len(enc.encode(s or ""))

def slice_by_tokens(s: str, start_tok: int, end_tok: int) -> str:
    ids = enc.encode(s or "")
    return enc.decode(ids[start_tok:end_tok])

def equal_token_chunks(text: str, target_chunk_tokens: int) -> List[str]:
    total = count_tokens(text)
    n_chunks = math.ceil(total / target_chunk_tokens)
    chunks = []
    for i in range(n_chunks):
        start_tok = i * target_chunk_tokens
        end_tok = min((i + 1) * target_chunk_tokens, total)
        chunks.append(slice_by_tokens(text, start_tok, end_tok))
    return chunks

def summarize_once(client: OpenAI, text: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=text
    )
    return response.output[0].content[0].text


def summarize(client, text: str) -> str:
    """
    If ≤ HALF_CTX_TOKENS, summarize directly.
    If exceeded, split evenly → summarize each chunk → resummarize the combined output."
    """
    n_tok = count_tokens(text)

    if n_tok <= HALF_CTX_TOKENS:
        return summarize_once(client, copy.deepcopy(PROMPT).format(text))

    chunks = equal_token_chunks(text, HALF_CTX_TOKENS)

    partial_summaries = [summarize_once(client, copy.deepcopy(PROMPT).format(ch)) for ch in chunks]

    joined = "\n\n".join(partial_summaries)
    final_summary = summarize_once(client, copy.deepcopy(PROMPT).format(joined))
    return final_summary

if "__name__" == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    client = OpenAI()

    with open(args.input_path, "r") as f:
        data = json.load(f)

    for row in tqdm.tqdm(data, total=len(data)):
        document = row['document']
        summary = summarize(client, document)
        row['summary'] = summary

    with open(args.output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)