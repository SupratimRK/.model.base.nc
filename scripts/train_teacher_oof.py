import os
import gc
import json
import time
import string
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CURATED_DIR = os.path.join(PROJECT_ROOT, "curated_datasets")

STAGE1_PATH = os.path.join(CURATED_DIR, "train_stage1_oof.parquet")
VAL_CURATED_PATH = os.path.join(CURATED_DIR, "val_curated.parquet")
OUT_CLEANED_PATH = os.path.join(CURATED_DIR, "train_cleaned.parquet")
OUT_CALIB_PATH = os.path.join(CURATED_DIR, "train_calib.parquet")
OUT_REPORT_PATH = os.path.join(CURATED_DIR, "clean_stage2_report.json")

CHAR_MAX_LEN = 32
# Character vocabulary: 0=PAD, 1=UNK, then printable ASCII lowercase + space
CHAR_LIST = list(" " + string.ascii_lowercase + string.digits + "-._'\"/:@#$%&*()+=[]{}|\\<>!?~`^,;")
CHAR_TO_ID = {c: i + 2 for i, c in enumerate(CHAR_LIST)}
CHAR_VOCAB_SIZE = len(CHAR_LIST) + 2


def encode_chars(texts: list[str], max_len: int = CHAR_MAX_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(texts)
    ids = np.zeros((n, max_len), dtype=np.uint8)
    masks = np.zeros((n, max_len), dtype=np.uint8)
    c2i = CHAR_TO_ID
    for i, t in enumerate(texts):
        L = min(len(t), max_len)
        for j in range(L):
            ids[i, j] = c2i.get(t[j], 1)
        if L > 0:
            masks[i, :L] = 1
    return torch.from_numpy(ids), torch.from_numpy(masks)


class CharTeacherNet(nn.Module):
    """
    Fast GTX 1650 Offline Teacher: Multi-Scale Dilated Char-CNN + Attention Pooling.
    Runs ~80s/epoch on 1.45M rows on GTX 1650 while having zero WordPiece blind spots.
    """
    def __init__(self, vocab_size=CHAR_VOCAB_SIZE, emb_dim=48, hidden_dim=128, max_len=CHAR_MAX_LEN):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.stem = nn.Conv1d(emb_dim, hidden_dim, kernel_size=3, padding=1)
        self.stem_norm = nn.LayerNorm(hidden_dim)
        self.conv_k3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv_d2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.attn_proj = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(0.15)
        self.classifier = nn.Linear(hidden_dim * 2, 2)
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, input_ids, attention_mask, return_aux=False):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.char_emb(input_ids.long()) + self.pos_emb(pos)  # [B, L, E]
        h = F.gelu(self.stem(x.transpose(1, 2)))                  # [B, H, L]
        h = h + F.gelu(self.conv_k3(h))
        h = h + F.gelu(self.conv_k5(h))
        h = h + F.gelu(self.conv_d2(h))

        ht = self.stem_norm(h.transpose(1, 2))                    # [B, L, H]
        pad_mask = (attention_mask == 0)
        attn_logits = self.attn_proj(ht).squeeze(-1).masked_fill(pad_mask, -1e4)
        attn_w = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        attn_p = (ht * attn_w).sum(dim=1)
        max_p = ht.masked_fill(pad_mask.unsqueeze(-1), -1e4).max(dim=1).values

        feat = self.drop(torch.cat([attn_p, max_p], dim=-1))
        logits = self.classifier(feat)
        if return_aux:
            return logits, self.gender_head(feat)
        return logits


def symmetric_ce_loss(logits, targets, weights, alpha=0.7, beta=0.3):
    """Noise-robust Symmetric Cross-Entropy (Wang et al., ICCV 2019)."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs).clamp(1e-6, 1.0 - 1e-6)
    ce = F.nll_loss(log_probs, targets, reduction="none")

    one_hot = F.one_hot(targets, num_classes=2).float().clamp(1e-4, 1.0)
    rce = -torch.sum(probs * torch.log(one_hot), dim=-1)
    loss = alpha * ce + beta * rce
    return (loss * weights).sum() / torch.clamp(weights.sum(), min=1e-6)


def main():
    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===============================================================", flush=True)
    print(f"Stage 2: 5-Fold Neural Teacher OOF & Cleaning on {device}", flush=True)
    if device.type == "cuda":
        print(f"--> GPU: {torch.cuda.get_device_name(0)}", flush=True)
        torch.backends.cudnn.benchmark = True
    print("===============================================================", flush=True)

    df = pd.read_parquet(STAGE1_PATH)
    n = len(df)
    print(f"--> Loaded {n:,} deduplicated train rows from {STAGE1_PATH}", flush=True)

    texts = df["text_norm"].tolist()
    y_init = df["y_init"].to_numpy(dtype=np.int64)
    has_gender = df["has_gender"].to_numpy(dtype=np.int8)
    has_conflict = df["has_conflict"].to_numpy(dtype=np.int8)
    is_alpha = df["is_alpha_space"].to_numpy(dtype=np.int8)
    n_words = df["n_words"].to_numpy(dtype=np.int8)
    oof_tfidf = df["oof_tfidf"].to_numpy(dtype=np.float32)

    g_map = {"MALE": 0, "FEMALE": 1}
    g_arr = np.array([g_map.get(str(g).upper(), -100) if g is not None else -100 for g in df["gender"]], dtype=np.int64)

    # Encode characters into compact uint8 tensors (~144 MB total for 1.8M rows)
    print("--> Encoding characters into uint8 tensors...", flush=True)
    t_enc = time.time()
    char_ids, char_masks = encode_chars(texts, CHAR_MAX_LEN)
    print(f"    Encoded in {time.time() - t_enc:.1f}s (RAM: {(char_ids.numel() + char_masks.numel()) / 1e6:.1f} MB)", flush=True)

    # Sample weights for Teacher: emphasize pure-alpha negative boundary, downweight ungendered conflicts & trivial digit noise
    base_weights = np.ones(n, dtype=np.float32)
    base_weights[(y_init == 0) & (is_alpha == 1)] = 2.8  # Counter 6:1 pure-alpha pos/neg imbalance
    base_weights[(y_init == 0) & (is_alpha == 0)] = 0.35 # Don't let digits dominate gradients
    base_weights[(y_init == 1) & (has_gender == 1)] = 1.0
    base_weights[(y_init == 1) & (has_gender == 0)] = 0.55
    base_weights[has_conflict == 1] *= 0.4

    y_tensor = torch.from_numpy(y_init)
    g_tensor = torch.from_numpy(g_arr)
    w_tensor = torch.from_numpy(base_weights)

    # Collect unique tokens to score them OOF for multi-word contamination detection
    unique_tokens = sorted({w for t in texts for w in t.split()})
    tok_to_idx = {w: i for i, w in enumerate(unique_tokens)}
    print(f"--> Unique whitespace tokens to score OOF: {len(unique_tokens):,}", flush=True)
    tok_ids, tok_masks = encode_chars(unique_tokens, CHAR_MAX_LEN)
    tok_loader = DataLoader(TensorDataset(tok_ids, tok_masks), batch_size=4096, shuffle=False)

    # 5-Fold Neural OOF
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    strat_key = df["y_init"].astype(str) + "_" + df["has_gender"].astype(str) + "_" + df["is_alpha_space"].astype(str)

    oof_neural = np.zeros(n, dtype=np.float32)
    tok_probs_sum = np.zeros(len(unique_tokens), dtype=np.float32)

    BATCH_SIZE = 4096
    EPOCHS = 2

    # Save raw OOF arrays immediately after 5-fold loop so re-runs / threshold tuning don't need to retrain 5 folds!
    oof_cache_path = os.path.join(CURATED_DIR, "train_oof_cache.parquet")
    if os.path.exists(oof_cache_path) and os.environ.get("REUSE_OOF_CACHE") == "1":
        print(f"--> Reusing cached OOF predictions from {oof_cache_path}", flush=True)
        cache_df = pd.read_parquet(oof_cache_path)
        oof_neural = cache_df["oof_neural"].to_numpy(dtype=np.float32)
        min_tok_oof = cache_df["min_tok_oof"].to_numpy(dtype=np.float32)
        tok_prob_mean = np.load(os.path.join(CURATED_DIR, "tok_prob_mean.npy"))
    else:
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), strat_key), start=1):
            f_t0 = time.time()
            # GPU-resident fast shuffled indexing (avoids slow per-item DataLoader CPU overhead on 1.45M rows)
            tr_ids_g = char_ids[tr_idx].to(device)
            tr_masks_g = char_masks[tr_idx].to(device)
            tr_y_g = y_tensor[tr_idx].to(device)
            tr_g_g = g_tensor[tr_idx].to(device)
            tr_w_g = w_tensor[tr_idx].to(device)
            n_tr = len(tr_idx)

            model = CharTeacherNet().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=0.02)
            n_batches = (n_tr + BATCH_SIZE - 1) // BATCH_SIZE
            total_steps = n_batches * EPOCHS
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
            scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
            crit_gender = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)

            for epoch in range(1, EPOCHS + 1):
                model.train()
                ep_loss = 0.0
                perm = torch.randperm(n_tr, device=device)
                for start in range(0, n_tr, BATCH_SIZE):
                    idx_b = perm[start:start + BATCH_SIZE]
                    b_ids = tr_ids_g[idx_b]
                    b_mask = tr_masks_g[idx_b]
                    b_y = tr_y_g[idx_b]
                    b_g = tr_g_g[idx_b]
                    b_w = tr_w_g[idx_b]

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                        logits, g_logits = model(b_ids, b_mask, return_aux=True)
                        loss_name = symmetric_ce_loss(logits.float(), b_y, b_w, alpha=0.7, beta=0.3)
                        g_mask = (b_y == 1) & (b_g >= 0)
                        loss_gen = crit_gender(g_logits[g_mask].float(), b_g[g_mask]) if g_mask.any() else 0.0
                        loss = loss_name + 0.15 * loss_gen

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() == scale_before:
                        scheduler.step()
                    ep_loss += float(loss.item())

            del tr_ids_g, tr_masks_g, tr_y_g, tr_g_g, tr_w_g

            # OOF validation predictions
            model.eval()
            va_preds = []
            va_ds = TensorDataset(char_ids[va_idx], char_masks[va_idx])
            va_loader = DataLoader(va_ds, batch_size=8192, shuffle=False)
            with torch.no_grad():
                for b_ids, b_mask in va_loader:
                    b_ids = b_ids.to(device, non_blocking=True)
                    b_mask = b_mask.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                        logits = model(b_ids, b_mask, return_aux=False)
                    p = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
                    va_preds.append(p)
            oof_va = np.concatenate(va_preds)
            oof_neural[va_idx] = oof_va

            # Accumulate token scores across folds (avg of 5 models)
            tok_preds = []
            with torch.no_grad():
                for b_ids, b_mask in tok_loader:
                    b_ids = b_ids.to(device, non_blocking=True)
                    b_mask = b_mask.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                        logits = model(b_ids, b_mask, return_aux=False)
                    tok_preds.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
            tok_probs_sum += np.concatenate(tok_preds)

            agree = float(((oof_va >= 0.5).astype(np.int64) == y_init[va_idx]).mean())
            print(f"    Fold {fold}/5 finished in {time.time() - f_t0:.1f}s | Raw agreement: {agree*100:.2f}%", flush=True)

            del model, optimizer, scheduler, scaler, va_ds, va_loader
            torch.cuda.empty_cache()
            gc.collect()

        tok_prob_mean = tok_probs_sum / 5.0
        del char_ids, char_masks, tok_ids, tok_masks
        gc.collect()

        # Map min token OOF score for each row
        print("--> Computing min-token neural score per sample...", flush=True)
        min_tok_oof = np.ones(n, dtype=np.float32)
        for i, t in enumerate(texts):
            ws = t.split()
            if ws:
                min_tok_oof[i] = min(float(tok_prob_mean[tok_to_idx[w]]) for w in ws)

        pd.DataFrame({"oof_neural": oof_neural, "min_tok_oof": min_tok_oof}).to_parquet(oof_cache_path, index=False)
        np.save(os.path.join(CURATED_DIR, "tok_prob_mean.npy"), tok_prob_mean)
        print(f"--> Cached OOF predictions to {oof_cache_path}", flush=True)

    # Combined OOF consensus score: favor the neural Char-CNN OOF (trained with balanced
    # pure-alpha weights and Symmetric CE) over linear TF-IDF (which over-memorized noisy 1-grams)
    p_oof = (0.80 * oof_neural + 0.20 * oof_tfidf).astype(np.float32)
    df["oof_neural"] = oof_neural
    df["min_tok_oof"] = min_tok_oof
    df["p_oof"] = p_oof

    # Confident Learning Cleaning & Soft Target Assignment
    # 1. Flip strings where the neural char-CNN OOF strongly rejects them OR where a multi-word
    #    string contains an ultra-high-confidence non-name token (min_tok_oof < 0.015 / 0.010).
    y_clean = y_init.copy()
    soft_target = p_oof.copy()
    sample_weight = np.ones(n, dtype=np.float32)

    flip_1_to_0 = (y_init == 1) & (
        ((has_gender == 0) & ((oof_neural < 0.30) | (min_tok_oof < 0.015))) |
        ((has_gender == 1) & ((oof_neural < 0.15) | (min_tok_oof < 0.010)))
    )
    uncertain_1 = (y_init == 1) & (~flip_1_to_0) & (
        (oof_neural < 0.55) | (min_tok_oof < 0.05) | (has_conflict == 1)
    )
    clean_1 = (y_init == 1) & (~flip_1_to_0) & (~uncertain_1)

    y_clean[flip_1_to_0] = 0
    soft_target[flip_1_to_0] = np.minimum(np.minimum(oof_neural[flip_1_to_0], min_tok_oof[flip_1_to_0] + 0.08), 0.15)
    sample_weight[flip_1_to_0] = 0.90

    soft_target[uncertain_1] = np.clip(0.35 + 0.50 * oof_neural[uncertain_1], 0.25, 0.85)
    sample_weight[uncertain_1] = 0.45

    soft_target[clean_1] = np.where(
        has_gender[clean_1] == 1,
        np.maximum(0.95, oof_neural[clean_1]),
        np.maximum(0.90, oof_neural[clean_1]),
    )
    sample_weight[clean_1] = np.where(has_gender[clean_1] == 1, 1.15, 0.95)

    # 2. Likely mislabeled or clean y_init=0
    flip_0_to_1 = (y_init == 0) & (is_alpha == 1) & (has_conflict == 1) & (oof_neural > 0.88) & (min_tok_oof > 0.70)
    uncertain_0 = (y_init == 0) & (~flip_0_to_1) & (is_alpha == 1) & ((oof_neural > 0.65) | (has_conflict == 1))
    clean_0 = (y_init == 0) & (~flip_0_to_1) & (~uncertain_0)

    y_clean[flip_0_to_1] = 1
    soft_target[flip_0_to_1] = oof_neural[flip_0_to_1]
    sample_weight[flip_0_to_1] = 0.70

    soft_target[uncertain_0] = np.clip(0.45 * oof_neural[uncertain_0], 0.10, 0.60)
    sample_weight[uncertain_0] = 0.50

    soft_target[clean_0] = np.minimum(0.08, oof_neural[clean_0])
    sample_weight[clean_0] = np.where(is_alpha[clean_0] == 1, 1.35, 0.80)

    df["y_clean"] = y_clean.astype(np.int64)
    df["soft_target"] = soft_target.astype(np.float32)
    df["sample_weight"] = sample_weight.astype(np.float32)
    df["is_synthetic"] = np.int8(0)

    print(f"    Flipped 1->0: {int(flip_1_to_0.sum()):,} | Uncertain 1: {int(uncertain_1.sum()):,} | Clean 1: {int(clean_1.sum()):,}", flush=True)
    print(f"    Flipped 0->1: {int(flip_0_to_1.sum()):,} | Uncertain 0: {int(uncertain_0.sum()):,} | Clean 0: {int(clean_0.sum()):,}", flush=True)

    # Carve out leak-free calibration split BEFORE adding synthetic negatives
    # Use high-confidence rows stratified across label + pure-alpha + word count
    rng = np.random.default_rng(42)
    high_conf_mask = (clean_1 | clean_0) & (sample_weight >= 0.85)
    hc_indices = np.where(high_conf_mask)[0]
    calib_size = min(40000, len(hc_indices) // 10)
    # Stratified sample from high_conf_mask
    hc_df = df.iloc[hc_indices][["y_clean", "is_alpha_space", "n_words"]].copy()
    hc_df["strat"] = hc_df["y_clean"].astype(str) + "_" + hc_df["is_alpha_space"].astype(str) + "_" + np.minimum(hc_df["n_words"], 2).astype(str)
    calib_idx = (
        hc_df.groupby("strat", group_keys=False)
        .apply(lambda g: g.sample(n=max(1, int(round(calib_size * len(g) / len(hc_df)))), random_state=42))
        .index.to_numpy()
    )
    calib_set = set(calib_idx.tolist())
    is_calib = np.array([i in calib_set for i in range(n)], dtype=bool)

    df_calib = df.loc[is_calib, ["text_norm", "y_clean", "soft_target", "gender", "is_alpha_space", "n_words"]].rename(
        columns={"text_norm": "text", "y_clean": "is_name"}
    ).reset_index(drop=True)
    df_train = df.loc[~is_calib].copy()

    # Synthesize hard negatives ONLY into df_train (never into df_calib)
    print("--> Synthesizing targeted pure-alpha & contaminated negatives from train tokens...", flush=True)
    # Use STRICT thresholds: only high-confidence single-word negatives (<0.01) and positives (>0.97)
    clean_real_single_tokens = [
        w for w, idx in tok_to_idx.items()
        if w.isalpha() and len(w) >= 3 and tok_prob_mean[idx] > 0.97
    ]
    clean_neg_single_tokens = [
        w for w, idx in tok_to_idx.items()
        if w.isalpha() and len(w) >= 3 and tok_prob_mean[idx] < 0.01
    ]
    clean_multi_names = df_train.loc[
        (df_train["y_clean"] == 1) & (df_train["sample_weight"] >= 1.0) & (df_train["n_words"] >= 2),
        "text_norm",
    ].tolist()

    synth_texts = []
    # A. 20,000 contaminated 2-word pure-alpha negatives (real_name + clear non_name_word)
    if clean_real_single_tokens and clean_neg_single_tokens:
        r_picks = rng.choice(clean_real_single_tokens, size=20000, replace=True)
        n_picks = rng.choice(clean_neg_single_tokens, size=20000, replace=True)
        for i in range(20000):
            if i % 2 == 0:
                synth_texts.append(f"{r_picks[i]} {n_picks[i]}")
            else:
                synth_texts.append(f"{n_picks[i]} {r_picks[i]}")

    # B. 10,000 non-name + non-name 2-word pure-alpha negatives
    if clean_neg_single_tokens:
        n1 = rng.choice(clean_neg_single_tokens, size=10000, replace=True)
        n2 = rng.choice(clean_neg_single_tokens, size=10000, replace=True)
        for a, b in zip(n1, n2):
            synth_texts.append(f"{a} {b}")

    # C. 20,000 noisy prefix/suffix/code handles wrapping real names
    dept_codes = ["cse", "ece", "it", "mech", "eee", "dev", "official", "gaming", "yt", "fps", "pro", "hq", "admin", "bot", "test"]
    if clean_multi_names and clean_real_single_tokens:
        m_picks = rng.choice(clean_multi_names, size=20000, replace=True)
        s_picks = rng.choice(clean_real_single_tokens, size=20000, replace=True)
        for i in range(20000):
            mode = i % 4
            if mode == 0:
                num = int(rng.integers(1, 9999))
                synth_texts.append(f"{num}_{m_picks[i]}")
            elif mode == 1:
                code = dept_codes[int(rng.integers(0, len(dept_codes)))]
                num = int(rng.integers(10, 9999))
                synth_texts.append(f"{s_picks[i]} {code}{num}")
            elif mode == 2:
                code = dept_codes[int(rng.integers(0, len(dept_codes)))]
                synth_texts.append(f"{s_picks[i]}_{code}")
            else:
                num = int(rng.integers(10, 2030))
                synth_texts.append(f"{m_picks[i]} {num}")

    # Remove any accidental collision with existing train or val texts
    existing_train = set(df_train["text_norm"].tolist())
    val_norms = set(
        pd.read_parquet(VAL_CURATED_PATH, columns=["text"])["text"]
        .dropna().astype(str).str.lower().str.strip().tolist()
    )
    synth_unique = []
    seen_s = set()
    for s in synth_texts:
        if s not in existing_train and s not in val_norms and s not in seen_s:
            seen_s.add(s)
            synth_unique.append(s)

    df_synth = pd.DataFrame({
        "text_norm": synth_unique,
        "gender": None,
        "has_gender": np.int8(0),
        "has_conflict": np.int8(0),
        "is_alpha_space": np.array([1 if all(c == " " or ("a" <= c <= "z") for c in s) else 0 for s in synth_unique], dtype=np.int8),
        "n_words": np.array([len(s.split()) for s in synth_unique], dtype=np.int8),
        "y_init": np.int8(0),
        "y_clean": np.int64(0),
        "soft_target": np.float32(0.02),
        "sample_weight": np.float32(1.0),
        "p_oof": np.float32(0.02),
        "oof_neural": np.float32(0.02),
        "oof_tfidf": np.float32(0.02),
        "min_tok_oof": np.float32(0.02),
        "is_synthetic": np.int8(1),
    })

    keep_cols = [
        "text_norm", "gender", "has_gender", "has_conflict", "is_alpha_space",
        "n_words", "y_init", "y_clean", "soft_target", "sample_weight",
        "p_oof", "oof_neural", "oof_tfidf", "min_tok_oof", "is_synthetic",
    ]
    df_out = pd.concat([df_train[keep_cols], df_synth[keep_cols]], ignore_index=True)
    df_out = df_out.rename(columns={"text_norm": "text", "y_clean": "is_name"})

    print(f"--> Saving cleaned train set ({len(df_out):,} rows) to {OUT_CLEANED_PATH}", flush=True)
    df_out.to_parquet(OUT_CLEANED_PATH, index=False)
    print(f"--> Saving leak-free calib set ({len(df_calib):,} rows) to {OUT_CALIB_PATH}", flush=True)
    df_calib.to_parquet(OUT_CALIB_PATH, index=False)

    report = {
        "input_rows": n,
        "flipped_1_to_0": int(flip_1_to_0.sum()),
        "uncertain_1": int(uncertain_1.sum()),
        "clean_1": int(clean_1.sum()),
        "flipped_0_to_1": int(flip_0_to_1.sum()),
        "uncertain_0": int(uncertain_0.sum()),
        "clean_0": int(clean_0.sum()),
        "calib_rows": len(df_calib),
        "synthetic_hard_negatives": len(df_synth),
        "final_train_rows": len(df_out),
        "final_train_pos": int((df_out["is_name"] == 1).sum()),
        "final_train_neg": int((df_out["is_name"] == 0).sum()),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(OUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("--> Stage 2 Report:", flush=True)
    print(json.dumps(report, indent=2), flush=True)

    print("\nSample 1->0 flips caught by Stage 2:")
    print(df.loc[flip_1_to_0, ["text_norm", "has_gender", "oof_neural", "oof_tfidf", "min_tok_oof"]].head(25).to_string())


if __name__ == "__main__":
    main()
