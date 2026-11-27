import os
import sys
import gc
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tokenizers import Tokenizer
from transformers import get_cosine_schedule_with_warmup

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
BATCH_SIZE = 2048
EPOCHS = 6
SWA_START_EPOCH = 4
LEARNING_RATE = 1.2e-3
WEIGHT_DECAY = 0.015
GENDER_LOSS_WEIGHT = 0.15
HARD_CE_WEIGHT = 0.40
SOFT_KL_WEIGHT = 0.60

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CURATED_DIR = os.path.join(PROJECT_ROOT, "curated_datasets")
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


def parse_gender_labels(series: pd.Series) -> np.ndarray:
    """Map MALE -> 0, FEMALE -> 1, all others -> -100 (ignored in loss)."""
    mapping = {"MALE": 0, "FEMALE": 1}
    str_series = series.fillna("").astype(str).str.upper()
    return np.array([mapping.get(g, -100) for g in str_series], dtype=np.int64)


def noise_robust_distill_loss(logits, hard_targets, soft_targets, weights):
    """
    Combines noise-robust Symmetric CE on hard labels with Soft-Target Distillation
    from the 5-fold Teacher OOF probabilities, weighted per sample.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs).clamp(1e-6, 1.0 - 1e-6)

    ce_hard = F.nll_loss(log_probs, hard_targets, reduction="none")
    one_hot = F.one_hot(hard_targets, num_classes=2).float().clamp(1e-4, 1.0)
    rce_hard = -torch.sum(probs * torch.log(one_hot), dim=-1)
    sce_hard = 0.75 * ce_hard + 0.25 * rce_hard

    p1 = soft_targets.clamp(1e-4, 1.0 - 1e-4)
    soft_dist = torch.stack([1.0 - p1, p1], dim=-1)
    ce_soft = -torch.sum(soft_dist * log_probs, dim=-1)

    total = HARD_CE_WEIGHT * sce_hard + SOFT_KL_WEIGHT * ce_soft
    return (total * weights).sum() / torch.clamp(weights.sum(), min=1e-6)


def fit_temperature_and_threshold(logits_np: np.ndarray, targets_np: np.ndarray):
    """
    Fit scalar temperature T > 0 on leak-free train_calib logits via NLL minimization,
    then sweep probability threshold tau maximizing balanced accuracy + F1.
    """
    logits_t = torch.tensor(logits_np, dtype=torch.float32)
    targets_t = torch.tensor(targets_np, dtype=torch.long)

    log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        temp = torch.exp(log_temp).clamp(0.2, 5.0)
        loss = F.cross_entropy(logits_t / temp, targets_t)
        loss.backward()
        return loss

    opt.step(closure)
    best_temp = float(torch.exp(log_temp).clamp(0.2, 5.0).item())
    scaled_probs = torch.softmax(logits_t / best_temp, dim=-1)[:, 1].numpy()

    best_tau = 0.50
    best_score = -1.0
    best_acc = 0.0
    best_f1 = 0.0
    for tau in np.linspace(0.30, 0.75, 91):
        preds = (scaled_probs >= tau).astype(np.int64)
        acc = accuracy_score(targets_np, preds)
        _, _, f1, _ = precision_recall_fscore_support(targets_np, preds, average="binary", zero_division=0)
        score = 0.5 * acc + 0.5 * f1
        if score > best_score:
            best_score = score
            best_tau = float(round(float(tau), 4))
            best_acc = float(acc)
            best_f1 = float(f1)

    return best_temp, best_tau, best_acc, best_f1


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("===============================================================")
    print("Stage 3: Training CompactNameClassifier (Distillation + SWA)")
    print("===============================================================")

    # 1. Load Rust Tokenizer
    tok_json_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
    print(f"--> Loading fast Rust tokenizer from: {tok_json_path}")
    tokenizer = Tokenizer.from_file(tok_json_path)
    tokenizer.enable_padding(length=MAX_LENGTH, pad_id=0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=MAX_LENGTH)

    # 2. Load Cleaned Train & Leak-Free Calibration Split (val_curated remains untouched)
    cleaned_train = os.path.join(CURATED_DIR, "train_cleaned.parquet")
    calib_path = os.path.join(CURATED_DIR, "train_calib.parquet")
    fallback_train = os.path.join(CURATED_DIR, "train_curated.parquet")

    if os.path.exists(cleaned_train) and os.path.exists(calib_path):
        print(f"--> Loading cleaned train dataset: {cleaned_train}")
        df_train = pd.read_parquet(cleaned_train)
        print(f"--> Loading leak-free calib split: {calib_path}")
        df_calib = pd.read_parquet(calib_path)
    else:
        print(f"--> Fallback loading {fallback_train}")
        df_full = pd.read_parquet(fallback_train).dropna(subset=["text", "is_name"]).copy()
        df_calib = df_full.sample(n=min(40000, len(df_full) // 10), random_state=42)
        df_train = df_full.drop(df_calib.index)
        df_train["soft_target"] = df_train["is_name"].astype(np.float32)
        df_train["sample_weight"] = np.float32(1.0)

    # Row-aligned dropna (fixes alignment bug)
    df_train = df_train.dropna(subset=["text", "is_name"]).copy()
    df_calib = df_calib.dropna(subset=["text", "is_name"]).copy()

    train_texts = df_train["text"].astype(str).str.lower().str.strip().tolist()
    train_labels = df_train["is_name"].astype(np.int64).to_numpy()
    train_soft = df_train["soft_target"].astype(np.float32).to_numpy() if "soft_target" in df_train.columns else train_labels.astype(np.float32)
    train_weights = df_train["sample_weight"].astype(np.float32).to_numpy() if "sample_weight" in df_train.columns else np.ones(len(df_train), dtype=np.float32)
    train_genders = parse_gender_labels(df_train["gender"]) if "gender" in df_train.columns else np.full(len(df_train), -100, dtype=np.int64)

    calib_texts = df_calib["text"].astype(str).str.lower().str.strip().tolist()
    calib_labels = df_calib["is_name"].astype(np.int64).to_numpy()

    del df_train, df_calib
    gc.collect()

    num_real = int((train_labels == 1).sum())
    num_non = int((train_labels == 0).sum())
    num_gendered = int((train_genders >= 0).sum())
    print(f"    Train size: {len(train_texts):,} (Real: {num_real:,}, Non: {num_non:,}, Gender Annotated: {num_gendered:,})")
    print(f"    Calib size: {len(calib_texts):,} (Real: {int((calib_labels==1).sum()):,}, Non: {int((calib_labels==0).sum()):,})")

    # 3. Pre-Tokenization
    train_ids, train_masks = pre_tokenize_dataset(tokenizer, train_texts, max_length=MAX_LENGTH)
    calib_ids, calib_masks = pre_tokenize_dataset(tokenizer, calib_texts, max_length=MAX_LENGTH)
    del train_texts, calib_texts
    gc.collect()

    train_labels_t = torch.from_numpy(train_labels)
    train_soft_t = torch.from_numpy(train_soft)
    train_weights_t = torch.from_numpy(train_weights)
    train_genders_t = torch.from_numpy(train_genders)
    calib_labels_t = torch.from_numpy(calib_labels)

    if device.type == "cuda":
        tr_ids_g = train_ids.to(device)
        tr_masks_g = train_masks.to(device)
        tr_labels_g = train_labels_t.to(device)
        tr_soft_g = train_soft_t.to(device)
        tr_weights_g = train_weights_t.to(device)
        tr_genders_g = train_genders_t.to(device)
    else:
        tr_ids_g = train_ids
        tr_masks_g = train_masks
        tr_labels_g = train_labels_t
        tr_soft_g = train_soft_t
        tr_weights_g = train_weights_t
        tr_genders_g = train_genders_t

    calib_loader = DataLoader(TensorDataset(calib_ids, calib_masks, calib_labels_t), batch_size=2048, shuffle=False)

    # 4. Initialize Model
    print("--> Initializing CompactNameClassifier (532k params + aux gender head)...")
    model = CompactNameClassifier(
        vocab_size=VOCAB_SIZE,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_len=MAX_LENGTH,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total Parameters (with aux head): {total_params:,}")

    # Mild class balancing (avoid aggressive over-flipping of real names)
    eff_pos = float(train_weights[train_labels == 1].sum())
    eff_neg = float(train_weights[train_labels == 0].sum())
    eff_tot = eff_pos + eff_neg
    w_neg_cls = min(max(eff_tot / (2.0 * max(1.0, eff_neg)), 0.85), 1.40)
    w_pos_cls = min(max(eff_tot / (2.0 * max(1.0, eff_pos)), 0.85), 1.40)
    sample_cls_mult = np.where(train_labels == 1, w_pos_cls, w_neg_cls).astype(np.float32)
    tr_weights_g = tr_weights_g * torch.from_numpy(sample_cls_mult).to(device)

    print(f"--> Class Multipliers: NON_NAME={w_neg_cls:.3f}, REAL_NAME={w_pos_cls:.3f}")

    n_tr = len(train_labels)
    n_batches = (n_tr + BATCH_SIZE - 1) // BATCH_SIZE
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = n_batches * EPOCHS
    warmup_steps = int(total_steps * 0.08)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion_gender = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    swa_state = None
    swa_count = 0
    best_calib_f1 = 0.0
    best_calib_acc = 0.0
    weights_path = os.path.join(SAVE_DIR, "model_weights.pt")

    print(f"\n--> Starting GPU training: {EPOCHS} epochs, batch_size={BATCH_SIZE}, SWA from epoch {SWA_START_EPOCH}", flush=True)
    start_total_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_name_loss = 0.0
        total_gen_loss = 0.0

        perm = torch.randperm(n_tr, device=device)
        for start in range(0, n_tr, BATCH_SIZE):
            idx_b = perm[start:start + BATCH_SIZE]
            b_ids = tr_ids_g[idx_b]
            b_mask = tr_masks_g[idx_b]
            b_names = tr_labels_g[idx_b]
            b_soft = tr_soft_g[idx_b]
            b_w = tr_weights_g[idx_b]
            b_genders = tr_genders_g[idx_b]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                name_logits, gender_logits = model(b_ids, b_mask, return_aux=True)
                loss_name = noise_robust_distill_loss(name_logits.float(), b_names, b_soft, b_w)

                valid_gender_mask = (b_names == 1) & (b_genders >= 0)
                if valid_gender_mask.any():
                    loss_gender = criterion_gender(gender_logits[valid_gender_mask].float(), b_genders[valid_gender_mask])
                else:
                    loss_gender = torch.tensor(0.0, device=device)

                loss = loss_name + (GENDER_LOSS_WEIGHT * loss_gender)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() == scale_before:
                scheduler.step()

            total_loss += float(loss.item())
            total_name_loss += float(loss_name.item())
            total_gen_loss += float(loss_gender.item())

        avg_train_loss = total_loss / n_batches
        avg_name_loss = total_name_loss / n_batches
        avg_gen_loss = total_gen_loss / n_batches

        if epoch >= SWA_START_EPOCH:
            cur_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state = cur_sd
            else:
                for k in swa_state:
                    swa_state[k] += cur_sd[k]
            swa_count += 1

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for b_ids, b_mask, b_lbls in calib_loader:
                b_ids = b_ids.to(device, non_blocking=True)
                b_mask = b_mask.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                    logits = model(b_ids, b_mask, return_aux=False)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(b_lbls.numpy())

        cal_acc = accuracy_score(all_targets, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary")
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:2d}/{EPOCHS} [{epoch_time:.1f}s] - "
            f"Loss: {avg_train_loss:.4f} (Name: {avg_name_loss:.4f}, Gen: {avg_gen_loss:.4f}) | "
            f"Calib Acc: {cal_acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%",
            flush=True,
        )

        if f1 > best_calib_f1:
            best_calib_f1 = f1
            best_calib_acc = cal_acc
            torch.save(model.state_dict(), weights_path)
            print(f"  --> Saved new best checkpoint (Calib F1: {f1*100:.2f}%, Acc: {cal_acc*100:.2f}%)", flush=True)

    if swa_state is not None and swa_count > 1:
        print(f"\n--> Evaluating SWA average of last {swa_count} epochs on train_calib...")
        for k in swa_state:
            if swa_state[k].is_floating_point():
                swa_state[k] = swa_state[k] / float(swa_count)
            else:
                swa_state[k] = swa_state[k] // swa_count
        model.load_state_dict({k: v.to(device) for k, v in swa_state.items()})
        model.eval()
        swa_preds = []
        swa_targets = []
        with torch.no_grad():
            for b_ids, b_mask, b_lbls in calib_loader:
                b_ids = b_ids.to(device, non_blocking=True)
                b_mask = b_mask.to(device, non_blocking=True)
                logits = model(b_ids, b_mask, return_aux=False)
                swa_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                swa_targets.extend(b_lbls.numpy())
        swa_acc = accuracy_score(swa_targets, swa_preds)
        _, _, swa_f1, _ = precision_recall_fscore_support(swa_targets, swa_preds, average="binary")
        print(f"    SWA Calib Acc: {swa_acc*100:.2f}% | SWA Calib F1: {swa_f1*100:.2f}%")
        if swa_f1 >= best_calib_f1 - 0.0005:
            best_calib_f1 = swa_f1
            best_calib_acc = swa_acc
            torch.save(model.state_dict(), weights_path)
            print("    --> Selected SWA checkpoint as final model_weights.pt!")
        else:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    else:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    print("\n--> Fitting post-training Temperature & Threshold on leak-free train_calib...")
    model.eval()
    calib_logits_list = []
    calib_targets_list = []
    with torch.no_grad():
        for b_ids, b_mask, b_lbls in calib_loader:
            b_ids = b_ids.to(device, non_blocking=True)
            b_mask = b_mask.to(device, non_blocking=True)
            logits = model(b_ids, b_mask, return_aux=False)
            calib_logits_list.append(logits.cpu().numpy())
            calib_targets_list.append(b_lbls.numpy())
    calib_logits_np = np.concatenate(calib_logits_list, axis=0)
    calib_targets_np = np.concatenate(calib_targets_list, axis=0)

    temperature, threshold, cal_opt_acc, cal_opt_f1 = fit_temperature_and_threshold(calib_logits_np, calib_targets_np)
    print(f"    Calibrated Temperature: {temperature:.4f}")
    print(f"    Calibrated Threshold:   {threshold:.4f} (Calib Acc: {cal_opt_acc*100:.2f}%, F1: {cal_opt_f1*100:.2f}%)")

    total_duration = time.time() - start_total_time
    print("\n===============================================================")
    print(f"TRAINING COMPLETE in {total_duration:.1f}s!")
    print(f"Final Calib Accuracy: {cal_opt_acc*100:.2f}% | F1: {cal_opt_f1*100:.2f}%")
    print(f"Weights saved to: {weights_path}")
    print("===============================================================")

    config = {
        "model_type": "CompactNameClassifier",
        "vocab_size": VOCAB_SIZE,
        "emb_dim": EMB_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "max_length": MAX_LENGTH,
        "lowercase": True,
        "labels": {"0": "NON_NAME", "1": "REAL_NAME"},
        "temperature": round(temperature, 4),
        "threshold": round(threshold, 4),
        "word_threshold": round(max(0.35, threshold - 0.12), 4),
        "best_calib_accuracy": float(cal_opt_acc),
        "best_calib_f1": float(cal_opt_f1),
        "training_scaffold": "5fold_char_teacher_distillation_sce_swa",
        "aux_loss_weight": GENDER_LOSS_WEIGHT,
    }
    config_path = os.path.join(CONFIG_DIR, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"--> [Saved] Configuration: {config_path}")

    try:
        from export_onnx import export_and_quantize
        print("\n--> [Auto-Export] Quantizing single-task graph to INT8 ONNX...")
        export_and_quantize()
    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")


if __name__ == "__main__":
    main()
