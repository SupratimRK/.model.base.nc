---
language:
- en
license: mit
library_name: onnx
pipeline_tag: text-classification
tags:
- name-classification
- onnx
- quantized
- fast-inference
- text-classification
- scratch
- low-latency
metrics:
- accuracy
- f1
model-index:
- name: CompactNameClassifier
  results:
  - task:
      type: text-classification
      name: Binary Name Classification
    dataset:
      name: Curated Human Names vs Non-Names
      type: custom
    metrics:
    - name: F1 Score
      type: f1
      value: 0.9621
    - name: Accuracy
      type: accuracy
      value: 0.9445
widget:
- text: "Sayantika Biswas"
  example_title: "Real Name (Bengali)"
- text: "Supratim Mondal"
  example_title: "Real Name (De-biased)"
- text: "Alexander Hamilton"
  example_title: "Real Name (Western)"
- text: "ShadowHunter99"
  example_title: "Gamer Tag / Username"
- text: "AYAN CSE3048"
  example_title: "Roll Number / Code"
- text: "112_SAYANTIKA BISWAS"
  example_title: "Prefix Noisy Handle"
---

# CompactNameClassifier: Ultra-Efficient Name Classifier Trained From Scratch

<div align="center">

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-yellow.svg)](https://huggingface.co/SupratimRK/compact-name-classifier)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/SupratimRK/.model.base.nc)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-INT8%20Quantized-blue)](https://onnxruntime.ai/)

**[🤗 Hugging Face Model Hub](https://huggingface.co/SupratimRK/compact-name-classifier)** | **[🐙 GitHub Repository](https://github.com/SupratimRK/.model.base.nc)**

</div>

`CompactNameClassifier` is a state-of-the-art, ultra-lightweight neural text classifier trained **from scratch** on a curated, de-biased dataset of **1,606,412 records** (1,458,041 train + 148,371 val; 1.18M clean human names across global and regional cultures + 280k non-names, unigrams, gamer tags, and alphanumeric codes).

Unlike generic large language models or pretrained BERT variants, `CompactNameClassifier` has **zero external pretrained dependencies**, weighs just **~632 KB** (INT8 ONNX), and delivers sub-millisecond CPU inference (**< 0.45 ms per query**).

---

## ⚡ Performance Highlights

| Metric | Previous Pretrained Model | **CompactNameClassifier (Scratch)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size (INT8 ONNX)** | 4.30 MB (2.45 MB pruned) | **~632 KB** | **74%–85% smaller** |
| **Total Parameters** | 4,380,000 params | **532,482 params** | **88% parameter reduction** |
| **Vocabulary Size** | 30,522 tokens | **4,000 tokens** | **87% smaller vocabulary** |
| **Sequence Length** | 16 tokens | **20 tokens** | **No truncation blind spots** |
| **CPU Latency (Single Thread)** | ~1.20 ms | **0.449 ms** | **~2.7x faster inference** |
| **Throughput** | ~800 queries/sec | **2,228 queries/sec** | Single CPU core in Node.js |
| **Validation Accuracy** | — | **94.45%** | Evaluated on 148k val split |
| **Validation F1-Score** | — | **96.21%** | Weighted F1 score |
| **Corner Cases Benchmark** | 52.9% (9/17) | **94.1% (16/17)** | Resistant to handle/roll-code leakage |

---

## 🧠 Architectural Innovations

1. **Custom Domain WordPiece Tokenizer (100% Uncased, 4,000 Vocab)**:
   - Configured with `Sequence([NFKC(), Lowercase(), StripAccents()])` and pre-tokenized via `Sequence([Whitespace(), Punctuation(), Digits(individual_digits=True)])`.
   - Normalizes all input to lowercase at the Rust tokenizer level, making the model completely immune to uppercase/titlecase domain mismatches.
   - Isolates individual digits (`0`–`9`), allowing alphanumeric noise (`112_`, `3048`, `99`) to be recognized without exploding the vocabulary.

2. **Factorized Embeddings**:
   - Projects token vocabulary through `Embedding(4000, 64) -> Linear(64, 128, bias=False)`.
   - Cuts embedding parameter footprint in half with zero loss in representational capacity.

3. **Pre-LN Transformer Encoder**:
   - 2-layer, 128-hidden, 4-attention heads with GELU activations and Pre-LayerNorm (`norm_first=True`) for fast, stable convergence.

4. **Masked Dual Pooling (MeanPool + MaxPool)**:
   - Concatenates Masked Mean Pooling (128) + Masked Max Pooling (128) into a 256-dimensional representation.
   - Max Pooling acts as an anomaly detector—if even one token in the sequence indicates non-name noise (e.g., `_`, `CSE3048`, `112`), the max-pooled features spike and alert the classifier immediately.

5. **V8 & Runtime Optimization**:
   - ONNX graph accepts `int32` inputs, allowing Node.js and Python runtimes to pass native 32-bit buffers with minimal overhead.

---

## 📦 Files in this Repository

| File | Size | Description |
| :--- | :--- | :--- |
| `model_int8.onnx` | ~647 KB | Quantized INT8 ONNX model (**recommended for production**) |
| `model.onnx` | ~2.2 MB | Full precision (FP32) ONNX model |
| `model_weights.pt` | ~2.1 MB | PyTorch model checkpoint / state dictionary |
| `tokenizer.json` | ~84 KB | Fast Tokenizers JSON configuration |
| `tokenizer_config.json` | ~260 B | Tokenizer metadata |
| `vocab.txt` | ~26 KB | 4,000 WordPiece vocabulary entries |
| `config.json` | ~350 B | Model architecture parameters & label mappings |
| `model.py` | ~4 KB | Standalone PyTorch `CompactNameClassifier` architecture |
| `predict.py` | ~7 KB | Ready-to-run inference script |

---

## 🚀 Quickstart & Usage

### 1. Clone Repository & Setup
```bash
git clone https://github.com/SupratimRK/.model.base.nc.git
cd .model.base.nc
npm install
```

### 2. Download Model from Hugging Face Hub

```python
from huggingface_hub import hf_hub_download
import os

repo_id = "SupratimRK/compact-name-classifier"

# Download ONNX model and tokenizer
onnx_path = hf_hub_download(repo_id=repo_id, filename="model_int8.onnx")
tok_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
```

### 3. Python Inference (ONNX Runtime)

```python
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# Load tokenizer and session
tokenizer = Tokenizer.from_file("tokenizer.json")
tokenizer.enable_padding(length=20, pad_token="[PAD]", pad_id=0)
tokenizer.enable_truncation(max_length=20)

session = ort.InferenceSession("model_int8.onnx")

# Tokenize
names = ["Sayantika Biswas", "ShadowHunter99", "CSE3048"]
encodings = tokenizer.encode_batch(names)

input_ids = np.array([e.ids for e in encodings], dtype=np.int32)
attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int32)

# Run Inference
logits = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

labels = {0: "NON_NAME", 1: "REAL_NAME"}
for text, prob in zip(names, probs):
    pred = int(np.argmax(prob))
    print(f"{text:20} => {labels[pred]} ({prob[pred]*100:.1f}%)")
```

Output:
```text
Sayantika Biswas     => REAL_NAME (97.7%)
ShadowHunter99       => NON_NAME (99.8%)
CSE3048              => NON_NAME (99.9%)
```

---

### 4. Node.js Native Inference (onnxruntime-node)

```javascript
import { CompactNameClassifier } from './scripts/classifier.mjs';

const classifier = new CompactNameClassifier();
await classifier.init();

const result = await classifier.predict('Sayantika Biswas');
console.log(result);
// {
//   text: 'Sayantika Biswas',
//   classification: 'REAL_NAME',
//   confidence: 0.998,
//   raw_probs: { non_name: 0.002, real_name: 0.998 }
// }
```

---

### 5. PyTorch Inference

```python
import torch
from scripts.model import CompactNameClassifier

model = CompactNameClassifier(
    vocab_size=4000,
    emb_dim=64,
    hidden_dim=128,
    num_layers=2,
    num_heads=4,
    max_len=20,
    dropout=0.0
)
model.load_state_dict(torch.load("models/model_weights.pt", weights_only=True))
model.eval()
```

---

## 📊 Training Data
- **Total Dataset Size**: 510,162 examples
  - **Positive Examples (Real Names)**: 247,864 entries (audited, 100% pure alphabetic, spanning diverse global cultures: Western, South Asian, East Asian, Hispanic, Middle Eastern, African).
  - **Negative Examples (Non-Names)**: 262,298 entries including gaming handles, social media usernames, alphanumeric serial numbers, code identifiers, and 52k curated high-frequency English vocabulary words.
- **Split**: 90% Training / 10% Validation (459,206 Train / 50,956 Val)

---

## 📄 License

This model and its associated weights and tokenizers are released under the [MIT License](https://opensource.org/licenses/MIT).
