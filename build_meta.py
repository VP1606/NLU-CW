"""
Build meta.pt from standard GloVe embeddings and vocab from trial data.
This allows inference without needing the original training meta.pt.
"""

import torch
import pickle
import numpy as np
from pathlib import Path
import os
import zipfile
import urllib.request
from tqdm import tqdm

# Create output directory
os.makedirs('./notebook_data', exist_ok=True)

print("=" * 60)
print("Building meta.pt from GloVe + trial data vocab")
print("=" * 60)

# Step 1: Download GloVe 6B 300d if not present
glove_file = './notebook_data/glove.6B.300d.txt'
if not os.path.exists(glove_file):
    print("\n[1/4] Downloading GloVe 6B 300d (~822MB)...")
    zip_path = f'{glove_file}.zip'
    url = 'https://nlp.stanford.edu/data/glove.6B.zip'

    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get('Content-Length') or 0)
        with open(zip_path, 'wb') as out, tqdm(
            total=total, unit='B', unit_scale=True, unit_divisor=1024,
            desc='glove.6B.zip',
        ) as bar:
            while True:
                chunk = resp.read(1024 * 64)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))

    print("      Extracting glove.6B.300d.txt...")
    with zipfile.ZipFile(zip_path) as zf:
        member = 'glove.6B.300d.txt'
        info = zf.getinfo(member)
        with zf.open(info) as src, open(glove_file, 'wb') as dst, tqdm(
            total=info.file_size, unit='B', unit_scale=True, unit_divisor=1024,
            desc=member,
        ) as bar:
            while True:
                chunk = src.read(1024 * 64)
                if not chunk:
                    break
                dst.write(chunk)
                bar.update(len(chunk))

    os.remove(zip_path)
    print("      Downloaded.")
else:
    print(f"\n[1/4] GloVe file exists: {glove_file}")

# Step 2: Load GloVe embeddings
print("\n[2/4] Loading GloVe embeddings...")
glove_embeddings = {}
total_bytes = os.path.getsize(glove_file)
with open(glove_file, 'r', encoding='utf-8') as f, tqdm(
    total=total_bytes, unit='B', unit_scale=True, unit_divisor=1024,
    desc='parsing GloVe',
) as bar:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        embedding = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        glove_embeddings[word] = embedding
        bar.update(len(line.encode('utf-8')))

print(f"      Loaded {len(glove_embeddings)} GloVe vectors")

# Step 3: Build vocab from trial data
print("\n[3/4] Building vocab from trial data...")
import pandas as pd
from collections import Counter

df = pd.read_csv('./data/NLI_trial.csv')
all_text = (df['premise'] + ' ' + df['hypothesis']).str.lower()

# Simple tokenization
words = []
for text in all_text:
    words.extend(text.split())

word_counts = Counter(words)
# Include words that appear in both trial data and GloVe
vocab = ['<PAD>', '<UNK>']  # PAD and UNK tokens
for word, count in word_counts.most_common(50000):
    if word in glove_embeddings:
        vocab.append(word)

print(f"      Vocab size: {len(vocab)}")

# Step 4: Build GloVe matrix and create meta.pt
print("\n[4/4] Creating GloVe matrix and char2idx, pos2idx...")

# GloVe matrix
glove_matrix = np.zeros((len(vocab), 300), dtype=np.float32)
for i, word in enumerate(vocab):
    if word in glove_embeddings:
        glove_matrix[i] = glove_embeddings[word]

# Character index (0-255 for ASCII)
char2idx = {chr(i): i for i in range(256)}

# POS tags (simplified)
pos2idx = {
    'NOUN': 0, 'VERB': 1, 'ADJ': 2, 'ADV': 3, 'PRON': 4, 'DET': 5,
    'ADP': 6, 'CCONJ': 7, 'SCONJ': 8, 'PART': 9, 'NUM': 10, 'INTJ': 11,
    'PROPN': 12, 'AUX': 13, 'PUNCT': 14, 'X': 15, 'SYM': 16, 'SPACE': 17,
}

# Save meta.pt
meta = {
    'vocab': vocab,
    'char2idx': char2idx,
    'pos2idx': pos2idx,
    'glove_matrix': glove_matrix,
}

output_path = './notebook_data/meta.pt'
torch.save(meta, output_path)
print(f"\nSaved meta.pt to {output_path}")
print(f"  vocab={len(vocab)}  char={len(char2idx)}  pos={len(pos2idx)}  glove_shape={glove_matrix.shape}")
print("\n✓ meta.pt ready for EmbeddingPrecomputer")
