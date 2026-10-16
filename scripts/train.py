import os
import sys
import time
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CompactNameClassifier

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
VOCAB_SIZE = 4000
EMB_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 4
MAX_LENGTH = 20
BATCH_SIZE = 256
EPOCHS = 6
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SCRATCH_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".model.scratch"))
MASTER_DATA_DIR = os.path.join(SCRATCH_DIR, "master_datasets")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "models", "tokenizer")
SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Device Configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"--> Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"--> Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
    print("--> Warning: CUDA not available, using CPU")

COMMON_KEYWORDS = {
    'get', 'set', 'post', 'put', 'delete', 'run', 'test', 'demo',
    'null', 'undefined', 'true', 'false', 'none', 'user', 'admin', 'root', 'api'
}

def augment_dataset(texts, labels, aug_rate=0.20):
    """
    Augments dataset with:
      1. Keyword correction: common verbs ('get', 'set') flagged as NON_NAME.
      2. Compound names: First + Last name pairs (e.g., 'Supratim Mondal') as REAL_NAME.
      3. Contaminated names: names polluted with roll codes/numbers ('112_Sayantika', 'Ayan CSE3048') as NON_NAME.
      4. Casing variations: lowercase, uppercase, titlecase.
    """
    random.seed(42)
    corrected_texts = []
    corrected_labels = []

    real_names = [str(t).strip() for t, l in zip(texts, labels) if l == 1 and str(t).strip().lower() not in COMMON_KEYWORDS]

    for text, label in zip(texts, labels):
        t_str = str(text).strip()
        if t_str.lower() in COMMON_KEYWORDS:
            corrected_texts.append(t_str)
            corrected_labels.append(0)
        else:
            corrected_texts.append(t_str)
            corrected_labels.append(label)

    # Compound Full Names (50,000 samples)
    print(f"--> Augmenting 50,000 compound full names (First + Last)...")
    sample_first = random.choices(real_names, k=50000)
    sample_last = random.choices(real_names, k=50000)
    for f, l in zip(sample_first, sample_last):
        compound = f"{f.title()} {l.title()}"
        corrected_texts.append(compound)
        corrected_labels.append(1)

    # Contaminated Noise Names (30,000 samples)
    print(f"--> Augmenting 30,000 contaminated names with roll numbers / dept codes...")
    noise_prefixes = ["112_", "101_", "cxd-", "roll_", "sec_", "01_"]
    noise_suffixes = ["_99", "CSE3048", "2024", "_fps", "123", "_official", "ECE01"]
    for name in random.choices(real_names, k=30000):
        r = random.random()
        if r < 0.4:
            noisy = random.choice(noise_prefixes) + name
        elif r < 0.8:
            noisy = name + " " + random.choice(noise_suffixes)
        else:
            noisy = name + random.choice(noise_suffixes)
        corrected_texts.append(noisy)
        corrected_labels.append(0)

    # Casing Augmentation
    final_texts = []
    for t, l in zip(corrected_texts, corrected_labels):
        if random.random() < aug_rate:
            r = random.random()
            if r < 0.4:
                final_texts.append(t.lower())
            elif r < 0.7:
                final_texts.append(t.upper())
            else:
                final_texts.append(t.title())
        else:
            final_texts.append(t)

    return final_texts, corrected_labels

def pre_tokenize_dataset(tokenizer, texts, max_length=MAX_LENGTH):
    """Pre-tokenize all texts in memory into contiguous int32 tensors."""
    t0 = time.time()
    print(f"--> Fast batch encoding {len(texts):,} strings into RAM...")
    enc = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(torch.int32)
    attention_mask = enc["attention_mask"].to(torch.int32)
    print(f"    Encoding completed in {time.time() - t0:.2f}s (Tensor size in RAM: {input_ids.element_size() * input_ids.nelement() / (1024**2):.1f} MB)")
    return input_ids, attention_mask

def main():
    print("===============================================================")
    print("Training Dedicated Ultra-Compact Name Classifier From Scratch")
    print("===============================================================")

    # 1. Load Tokenizer
    print(f"--> Loading custom tokenizer from: {TOKENIZER_DIR}")
    tok_json_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tok_json_path,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )

    # 2. Load Parquet Datasets
    train_parquet = os.path.join(MASTER_DATA_DIR, "train.parquet")
    val_parquet = os.path.join(MASTER_DATA_DIR, "val.parquet")

    print(f"--> Loading train data: {train_parquet}")
    df_train = pd.read_parquet(train_parquet)
    print(f"--> Loading val data:   {val_parquet}")
    df_val = pd.read_parquet(val_parquet)

    train_texts = df_train["text"].astype(str).tolist()
    train_labels = df_train["is_name"].astype(int).tolist()
    val_texts = df_val["text"].astype(str).tolist()
    val_labels = df_val["is_name"].astype(int).tolist()

    print(f"    Train size: {len(train_texts):,} (Real: {sum(train_labels):,}, Non: {len(train_labels)-sum(train_labels):,})")
    print(f"    Val size:   {len(val_texts):,} (Real: {sum(val_labels):,}, Non: {len(val_labels)-sum(val_labels):,})")

    # 3. Apply Multi-Task Real-World Augmentation (Compound names, noise injection, keywords)
    train_texts_aug, train_labels_aug = augment_dataset(train_texts, train_labels, aug_rate=0.20)
    print(f"    Augmented train size: {len(train_texts_aug):,} (Real: {sum(train_labels_aug):,}, Non: {len(train_labels_aug)-sum(train_labels_aug):,})")

    # 4. In-Memory Pre-Tokenization
    train_ids, train_masks = pre_tokenize_dataset(tokenizer, train_texts_aug, max_length=MAX_LENGTH)
    val_ids, val_masks = pre_tokenize_dataset(tokenizer, val_texts, max_length=MAX_LENGTH)

    train_labels_tensor = torch.tensor(train_labels_aug, dtype=torch.long)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_ids, train_masks, train_labels_tensor)
    val_dataset = TensorDataset(val_ids, val_masks, val_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, pin_memory=True)

    # 5. Initialize Model
    print("--> Initializing CompactNameClassifier architecture...")
    model = CompactNameClassifier(
        vocab_size=VOCAB_SIZE,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_len=MAX_LENGTH,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total Parameters:     {total_params:,}")
    print(f"    Trainable Parameters: {trainable_params:,}")

    # 6. Optimizer, Scheduler, Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 7. Training Loop
    best_val_f1 = 0.0
    best_val_acc = 0.0
    weights_path = os.path.join(SAVE_DIR, "model_weights.pt")

    print(f"\n--> Starting GPU training: {EPOCHS} epochs, batch_size={BATCH_SIZE}, mixed-precision FP16")
    start_total_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for b_ids, b_mask, b_lbls in train_loader:
            b_ids = b_ids.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)
            b_lbls = b_lbls.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(b_ids, b_mask)
                loss = criterion(logits, b_lbls)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for b_ids, b_mask, b_lbls in val_loader:
                b_ids = b_ids.to(device, non_blocking=True)
                b_mask = b_mask.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(b_ids, b_mask)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(b_lbls.numpy())

        val_acc = accuracy_score(all_targets, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary")
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{EPOCHS} [{epoch_time:.1f}s] - "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val Precision: {prec*100:.2f}% | "
            f"Val Recall: {rec*100:.2f}% | "
            f"Val F1: {f1*100:.2f}%"
        )

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), weights_path)
            print(f"  --> Saved new best checkpoint (F1: {f1*100:.2f}%, Acc: {val_acc*100:.2f}%)")

    total_duration = time.time() - start_total_time
    print("\n===============================================================")
    print(f"TRAINING COMPLETE in {total_duration:.1f}s!")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Best Validation F1 Score: {best_val_f1*100:.2f}%")
    print(f"Weights saved to: {weights_path}")
    print("===============================================================")

    # 8. Save Model Configuration JSON
    config = {
        "model_type": "CompactNameClassifier",
        "vocab_size": VOCAB_SIZE,
        "emb_dim": EMB_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "max_length": MAX_LENGTH,
        "labels": {0: "NON_NAME", 1: "REAL_NAME"},
        "best_val_accuracy": float(best_val_acc),
        "best_val_f1": float(best_val_f1)
    }
    config_path = os.path.join(CONFIG_DIR, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"--> [Saved] Configuration: {config_path}")

if __name__ == "__main__":
    main()
