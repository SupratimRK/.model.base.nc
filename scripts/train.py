import os
import sys
import gc
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tokenizers import Tokenizer
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast

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
BATCH_SIZE = 512
EPOCHS = 6
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
GENDER_LOSS_WEIGHT = 0.15  # Auxiliary regularizer weight recommended by Hydrogen

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

def pre_tokenize_dataset(tok, texts, max_length=MAX_LENGTH, chunk_size=100000):
    """Fast batch tokenization using Rust tokenizers directly into contiguous int32 tensors."""
    t0 = time.time()
    print(f"--> Fast batch encoding {len(texts):,} strings into RAM (chunks of {chunk_size:,})...")
    all_input_ids = []
    all_attention_mask = []

    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        enc = tok.encode_batch(chunk)
        ids = np.array([e.ids for e in enc], dtype=np.int32)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int32)
        all_input_ids.append(torch.from_numpy(ids))
        all_attention_mask.append(torch.from_numpy(mask))

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_mask, dim=0)
    del all_input_ids, all_attention_mask
    gc.collect()

    print(f"    Encoding completed in {time.time() - t0:.2f}s (Tensor size in RAM: {input_ids.element_size() * input_ids.nelement() / (1024**2):.1f} MB)")
    return input_ids, attention_mask

def parse_gender_labels(series: pd.Series) -> list:
    """Map MALE -> 0, FEMALE -> 1, all others -> -100 (ignored in loss)."""
    mapping = {"MALE": 0, "FEMALE": 1}
    str_series = series.fillna("").astype(str).str.upper()
    return [mapping.get(g, -100) for g in str_series]

def main():
    print("===============================================================")
    print("Training CompactNameClassifier (Single-Task with Gender Signal)")
    print("===============================================================")

    # 1. Load Rust Tokenizer
    tok_json_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    print(f"--> Loading fast Rust tokenizer from: {tok_json_path}")
    tokenizer = Tokenizer.from_file(tok_json_path)
    tokenizer.enable_padding(length=MAX_LENGTH, pad_id=0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=MAX_LENGTH)

    # 2. Load Parquet Datasets (Using Curated if available, else Master)
    default_curated_train = os.path.join(PROJECT_ROOT, "curated_datasets", "train_curated.parquet")
    default_curated_val = os.path.join(PROJECT_ROOT, "curated_datasets", "val_curated.parquet")

    train_parquet = default_curated_train if os.path.exists(default_curated_train) else os.path.join(MASTER_DATA_DIR, "train.parquet")
    val_parquet = default_curated_val if os.path.exists(default_curated_val) else os.path.join(MASTER_DATA_DIR, "val.parquet")

    print(f"--> Loading train dataset: {train_parquet}")
    df_train = pd.read_parquet(train_parquet)
    print(f"--> Loading val dataset:   {val_parquet}")
    df_val = pd.read_parquet(val_parquet)

    train_texts = df_train["text"].dropna().astype(str).str.lower().str.strip().tolist()
    train_labels = df_train["is_name"].astype(int).tolist()
    train_genders = parse_gender_labels(df_train["gender"]) if "gender" in df_train.columns else [-100] * len(train_labels)

    val_texts = df_val["text"].dropna().astype(str).str.lower().str.strip().tolist()
    val_labels = df_val["is_name"].astype(int).tolist()

    del df_train, df_val
    gc.collect()

    num_real = sum(train_labels)
    num_non = len(train_labels) - num_real
    num_gendered = sum(1 for g in train_genders if g >= 0)
    print(f"    Train size: {len(train_texts):,} (Real: {num_real:,}, Non: {num_non:,}, Gender Annotated: {num_gendered:,})")
    print(f"    Val size:   {len(val_texts):,} (Real: {sum(val_labels):,}, Non: {len(val_labels)-sum(val_labels):,})")

    # 3. In-Memory Pre-Tokenization
    train_ids, train_masks = pre_tokenize_dataset(tokenizer, train_texts, max_length=MAX_LENGTH)
    val_ids, val_masks = pre_tokenize_dataset(tokenizer, val_texts, max_length=MAX_LENGTH)
    del train_texts, val_texts
    gc.collect()

    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    train_genders_tensor = torch.tensor(train_genders, dtype=torch.long)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_ids, train_masks, train_labels_tensor, train_genders_tensor)
    val_dataset = TensorDataset(val_ids, val_masks, val_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, pin_memory=True)

    # 4. Initialize Model with Auxiliary Training Head
    print("--> Initializing CompactNameClassifier with auxiliary gender regularizer head...")
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
    print(f"    Total Parameters (with aux head): {total_params:,}")

    # 5. Class Weighted Loss & Auxiliary Criterion
    total = len(train_labels)
    weight_non = total / (2.0 * max(1, num_non))
    weight_real = total / (2.0 * max(1, num_real))
    class_weights = torch.tensor([weight_non, weight_real], dtype=torch.float, device=device)
    print(f"--> Loss Class Weights: NON_NAME={weight_non:.3f}, REAL_NAME={weight_real:.3f}")
    print(f"--> Auxiliary Gender Loss Weight lambda={GENDER_LOSS_WEIGHT} (masked strictly to valid real names)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    criterion_name = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    criterion_gender = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # 6. Training Loop
    best_val_f1 = 0.0
    best_val_acc = 0.0
    weights_path = os.path.join(SAVE_DIR, "model_weights.pt")

    print(f"\n--> Starting GPU training: {EPOCHS} epochs, batch_size={BATCH_SIZE}, mixed-precision FP16")
    start_total_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_name_loss = 0.0
        total_gen_loss = 0.0

        for b_ids, b_mask, b_names, b_genders in train_loader:
            b_ids = b_ids.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)
            b_names = b_names.to(device, non_blocking=True)
            b_genders = b_genders.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                name_logits, gender_logits = model(b_ids, b_mask, return_aux=True)

                loss_name = criterion_name(name_logits, b_names)

                # Strict Masking: gender loss applies ONLY to real names with annotated gender
                valid_gender_mask = (b_names == 1) & (b_genders >= 0)
                if valid_gender_mask.any():
                    loss_gender = criterion_gender(gender_logits[valid_gender_mask], b_genders[valid_gender_mask])
                else:
                    loss_gender = torch.tensor(0.0, device=device)

                loss = loss_name + (GENDER_LOSS_WEIGHT * loss_gender)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            total_name_loss += loss_name.item()
            total_gen_loss += loss_gender.item() if isinstance(loss_gender, torch.Tensor) else 0.0

        avg_train_loss = total_loss / len(train_loader)
        avg_name_loss = total_name_loss / len(train_loader)
        avg_gen_loss = total_gen_loss / len(train_loader)

        # 7. Single-Task Validation (evaluating strictly on is_name)
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for b_ids, b_mask, b_lbls in val_loader:
                b_ids = b_ids.to(device, non_blocking=True)
                b_mask = b_mask.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                    # Inference mode: single-task forward pass
                    logits = model(b_ids, b_mask, return_aux=False)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(b_lbls.numpy())

        val_acc = accuracy_score(all_targets, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary")
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{EPOCHS} [{epoch_time:.1f}s] - "
            f"Train Loss: {avg_train_loss:.4f} (Name: {avg_name_loss:.4f}, Gen: {avg_gen_loss:.4f}) | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val Prec: {prec*100:.2f}% | "
            f"Val Rec: {rec*100:.2f}% | "
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
        "lowercase": True,
        "labels": {0: "NON_NAME", 1: "REAL_NAME"},
        "best_val_accuracy": float(best_val_acc),
        "best_val_f1": float(best_val_f1),
        "training_scaffold": "auxiliary_gender_regularizer",
        "aux_loss_weight": GENDER_LOSS_WEIGHT
    }
    config_path = os.path.join(CONFIG_DIR, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"--> [Saved] Configuration: {config_path}")

    # 9. Auto-Export Single-Task INT8 ONNX Model
    try:
        from export_onnx import export_and_quantize
        print("\n--> [Auto-Export] Quantizing single-task graph to INT8 ONNX...")
        export_and_quantize()
    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")

if __name__ == "__main__":
    main()
