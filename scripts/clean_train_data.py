import os
import re
import gc
import json
import time
import unicodedata
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CURATED_DIR = os.path.join(PROJECT_ROOT, "curated_datasets")

TRAIN_CURATED_PATH = os.path.join(CURATED_DIR, "train_curated.parquet")
VAL_CURATED_PATH = os.path.join(CURATED_DIR, "val_curated.parquet")
OUT_STAGE1_PATH = os.path.join(CURATED_DIR, "train_stage1_oof.parquet")
OUT_REPORT_PATH = os.path.join(CURATED_DIR, "clean_stage1_report.json")

VOWELS = set("aeiouy")


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_lexical_features(texts: list[str]) -> np.ndarray:
    """Compute fast deterministic lexical features for OOF linear noise detector."""
    n = len(texts)
    feats = np.zeros((n, 11), dtype=np.float32)
    for i, t in enumerate(texts):
        length = len(t)
        words = t.split() if t else []
        n_words = len(words)
        alpha_cnt = sum(1 for c in t if "a" <= c <= "z")
        digit_cnt = sum(1 for c in t if "0" <= c <= "9")
        punct_cnt = sum(1 for c in t if not ("a" <= c <= "z" or "0" <= c <= "9" or c == " "))
        vowel_cnt = sum(1 for c in t if c in VOWELS)

        max_cons_run = 0
        cur_cons = 0
        max_repeat = 1
        cur_repeat = 1
        prev_c = ""
        for c in t:
            if "a" <= c <= "z" and c not in VOWELS:
                cur_cons += 1
                if cur_cons > max_cons_run:
                    max_cons_run = cur_cons
            else:
                cur_cons = 0
            if c == prev_c and c != " ":
                cur_repeat += 1
                if cur_repeat > max_repeat:
                    max_repeat = cur_repeat
            else:
                cur_repeat = 1
            prev_c = c

        min_w_len = min((len(w) for w in words), default=0)
        max_w_len = max((len(w) for w in words), default=0)
        min_w_vowel_ratio = min(
            (sum(1 for c in w if c in VOWELS) / max(1, len(w)) for w in words),
            default=0.0,
        )

        feats[i, 0] = min(length, 40) / 40.0
        feats[i, 1] = min(n_words, 5) / 5.0
        feats[i, 2] = vowel_cnt / max(1, alpha_cnt)
        feats[i, 3] = 1.0 if digit_cnt > 0 else 0.0
        feats[i, 4] = 1.0 if punct_cnt > 0 else 0.0
        feats[i, 5] = min(digit_cnt + punct_cnt, 10) / 10.0
        feats[i, 6] = min(max_cons_run, 10) / 10.0
        feats[i, 7] = min(max_repeat, 5) / 5.0
        feats[i, 8] = min(min_w_len, 15) / 15.0
        feats[i, 9] = min(max_w_len, 20) / 20.0
        feats[i, 10] = min_w_vowel_ratio
    return feats


def compute_fold_token_features(
    texts: list[str],
    y_train: np.ndarray,
    has_gender_train: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute leak-free intra-fold per-token statistics on tr_idx and map to both tr_idx and va_idx.
    Features per sample:
      [min_tok_pos_rate, max_tok_neg_only_flag, min_tok_gender_rate, any_tok_single_neg_only]
    """
    tok_pos = {}
    tok_neg = {}
    tok_gender = {}
    tok_single_pos = {}
    tok_single_neg = {}

    for idx in tr_idx:
        t = texts[idx]
        y = int(y_train[idx])
        hg = int(has_gender_train[idx])
        words = t.split()
        is_single = len(words) == 1
        for w in words:
            if y == 1:
                tok_pos[w] = tok_pos.get(w, 0) + 1
                if is_single:
                    tok_single_pos[w] = tok_single_pos.get(w, 0) + 1
            else:
                tok_neg[w] = tok_neg.get(w, 0) + 1
                if is_single:
                    tok_single_neg[w] = tok_single_neg.get(w, 0) + 1
            if hg:
                tok_gender[w] = tok_gender.get(w, 0) + 1

    def map_indices(indices: np.ndarray) -> np.ndarray:
        out = np.zeros((len(indices), 4), dtype=np.float32)
        for row_i, idx in enumerate(indices):
            words = texts[idx].split()
            if not words:
                continue
            min_pos_rate = 1.0
            max_neg_only = 0.0
            min_gen_rate = 1.0
            any_single_neg_only = 0.0
            for w in words:
                p = tok_pos.get(w, 0)
                n = tok_neg.get(w, 0)
                tot = p + n
                # Laplace smoothed positive rate (neutral 0.5 for unseen)
                pos_r = (p + 1.0) / (tot + 2.0)
                gen_r = (tok_gender.get(w, 0) + 0.5) / (tot + 2.0)
                if pos_r < min_pos_rate:
                    min_pos_rate = pos_r
                if gen_r < min_gen_rate:
                    min_gen_rate = gen_r
                if tot >= 2 and p == 0 and n >= 2:
                    max_neg_only = 1.0
                sp = tok_single_pos.get(w, 0)
                sn = tok_single_neg.get(w, 0)
                if sn >= 1 and sp == 0:
                    any_single_neg_only = 1.0
            out[row_i, 0] = min_pos_rate
            out[row_i, 1] = max_neg_only
            out[row_i, 2] = min_gen_rate
            out[row_i, 3] = any_single_neg_only
        return out

    return map_indices(tr_idx), map_indices(va_idx)


def main():
    t0 = time.time()
    print("===============================================================")
    print("Stage 1: Leak-Free Train Hygiene & 5-Fold Lexical OOF Scoring")
    print("===============================================================")

    # 1. Load datasets (freeze val_curated strictly for overlap removal only)
    print(f"--> Loading {TRAIN_CURATED_PATH}")
    df_tr = pd.read_parquet(
        TRAIN_CURATED_PATH,
        columns=["text", "is_name", "gender", "audit_status"],
    )
    print(f"--> Loading {VAL_CURATED_PATH} (for overlap removal only)")
    df_va = pd.read_parquet(VAL_CURATED_PATH, columns=["text"])

    raw_train_rows = len(df_tr)
    df_tr = df_tr.dropna(subset=["text", "is_name"]).copy()
    df_tr["text_norm"] = [normalize_text(t) for t in df_tr["text"].astype(str)]
    df_tr = df_tr[df_tr["text_norm"].str.len() > 0].copy()

    val_norms = set(normalize_text(t) for t in df_va["text"].dropna().astype(str))
    del df_va

    overlap_mask = df_tr["text_norm"].isin(val_norms)
    overlap_rows = int(overlap_mask.sum())
    overlap_unique = int(df_tr.loc[overlap_mask, "text_norm"].nunique())
    df_tr = df_tr[~overlap_mask].copy()
    print(f"    Removed {overlap_rows:,} train rows ({overlap_unique:,} unique texts) overlapping val_curated.")

    # 2. Deduplicate & aggregate conflict statistics
    print("--> Deduplicating & aggregating label/gender consensus...")
    df_tr["is_name"] = df_tr["is_name"].astype(np.int64)
    g_upper = df_tr["gender"].fillna("").astype(str).str.upper()
    df_tr["is_male"] = (g_upper == "MALE").astype(np.int64)
    df_tr["is_female"] = (g_upper == "FEMALE").astype(np.int64)

    agg = df_tr.groupby("text_norm", sort=False).agg(
        row_count=("is_name", "count"),
        pos_count=("is_name", "sum"),
        male_count=("is_male", "sum"),
        female_count=("is_female", "sum"),
    ).reset_index()
    del df_tr

    agg["neg_count"] = agg["row_count"] - agg["pos_count"]
    agg["raw_pos_rate"] = (agg["pos_count"] / agg["row_count"]).astype(np.float32)
    agg["has_conflict"] = ((agg["pos_count"] > 0) & (agg["neg_count"] > 0)).astype(np.int8)

    # Initial label: strict majority; ties go to 0 (conservative against false positives)
    agg["y_init"] = (agg["pos_count"] > agg["neg_count"]).astype(np.int8)

    # Consensus gender (only if valid gender present and non-contradictory)
    genders = np.full(len(agg), None, dtype=object)
    m_mask = (agg["male_count"] > 0) & (agg["female_count"] == 0) & (agg["y_init"] == 1)
    f_mask = (agg["female_count"] > 0) & (agg["male_count"] == 0) & (agg["y_init"] == 1)
    genders[m_mask.values] = "MALE"
    genders[f_mask.values] = "FEMALE"
    agg["gender"] = genders
    agg["has_gender"] = ((m_mask | f_mask)).astype(np.int8)

    unique_rows = len(agg)
    conflict_count = int(agg["has_conflict"].sum())
    print(f"    Unique non-val texts: {unique_rows:,} (collapsed {raw_train_rows - overlap_rows - unique_rows:,} duplicates)")
    print(f"    Conflicting texts resolved: {conflict_count:,}")
    print(f"    y_init distribution: 1={int((agg['y_init']==1).sum()):,}, 0={int((agg['y_init']==0).sum()):,}")
    print(f"    has_gender=1: {int(agg['has_gender'].sum()):,}")

    texts = agg["text_norm"].tolist()
    y_init = agg["y_init"].to_numpy(dtype=np.int8)
    has_gender = agg["has_gender"].to_numpy(dtype=np.int8)

    # 3. Extract static lexical features + global token contamination metadata (for diagnostics)
    print("--> Extracting lexical features...")
    t_feat = time.time()
    lex_feats = extract_lexical_features(texts)
    is_alpha_space = (lex_feats[:, 3] == 0.0) & (lex_feats[:, 4] == 0.0)
    n_words = np.array([len(t.split()) for t in texts], dtype=np.int8)
    print(f"    Lexical features extracted in {time.time() - t_feat:.1f}s")

    # 4. 5-Fold Out-of-Fold TF-IDF (char_wb + word) + Lexical + Fold Token Stats
    print("--> Running 5-Fold CPU TF-IDF (char_wb 2-5 + word 1-gram) OOF Confident Learning...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    strat_key = y_init.astype(str) + "_" + has_gender.astype(str) + "_" + is_alpha_space.astype(str)

    oof_tfidf = np.zeros(unique_rows, dtype=np.float32)
    oof_tok_min_pos_rate = np.zeros(unique_rows, dtype=np.float32)
    oof_tok_any_single_neg = np.zeros(unique_rows, dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(texts, strat_key), start=1):
        f_t0 = time.time()
        tr_texts = [texts[i] for i in tr_idx]
        va_texts = [texts[i] for i in va_idx]
        y_tr = y_init[tr_idx]

        vec_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=5,
            max_features=80000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        vec_word = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 1),
            min_df=5,
            max_features=30000,
            sublinear_tf=True,
            dtype=np.float32,
        )

        X_tr_char = vec_char.fit_transform(tr_texts)
        X_va_char = vec_char.transform(va_texts)

        X_tr_word = vec_word.fit_transform(tr_texts)
        X_va_word = vec_word.transform(va_texts)

        tr_tok_f, va_tok_f = compute_fold_token_features(texts, y_init, has_gender, tr_idx, va_idx)
        oof_tok_min_pos_rate[va_idx] = va_tok_f[:, 0]
        oof_tok_any_single_neg[va_idx] = va_tok_f[:, 3]

        scaler = StandardScaler()
        X_tr_dense = scaler.fit_transform(np.hstack([lex_feats[tr_idx], tr_tok_f])).astype(np.float32)
        X_va_dense = scaler.transform(np.hstack([lex_feats[va_idx], va_tok_f])).astype(np.float32)

        X_tr = sparse.hstack([X_tr_char, X_tr_word, sparse.csr_matrix(X_tr_dense)], format="csr")
        X_va = sparse.hstack([X_va_char, X_va_word, sparse.csr_matrix(X_va_dense)], format="csr")
        del X_tr_char, X_tr_word, X_va_char, X_va_word, X_tr_dense, X_va_dense, tr_tok_f, va_tok_f
        gc.collect()

        # Downweight conflicting rows so linear OOF resists label noise
        sw = np.ones(len(tr_idx), dtype=np.float32)
        sw[agg["has_conflict"].to_numpy()[tr_idx] == 1] = 0.5

        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            max_iter=20,
            early_stopping=True,
            validation_fraction=0.05,
            n_iter_no_change=3,
            tol=1e-4,
            random_state=42 + fold,
            n_jobs=1,
        )
        clf.fit(X_tr, y_tr, sample_weight=sw)
        probs = clf.predict_proba(X_va)[:, 1].astype(np.float32)
        oof_tfidf[va_idx] = probs

        del X_tr, X_va, vec_char, vec_word, clf, sw, tr_texts, va_texts
        gc.collect()

        fold_acc = float(((probs >= 0.5).astype(np.int8) == y_init[va_idx]).mean())
        print(f"    Fold {fold}/5 done in {time.time() - f_t0:.1f}s | Raw agreement: {fold_acc*100:.2f}%", flush=True)

    agg["is_alpha_space"] = is_alpha_space.astype(np.int8)
    agg["n_words"] = n_words
    agg["oof_tfidf"] = oof_tfidf
    agg["oof_tok_min_pos_rate"] = oof_tok_min_pos_rate
    agg["oof_tok_any_single_neg"] = oof_tok_any_single_neg

    # 5. Save Stage 1 OOF artifact
    print(f"--> Saving Stage 1 OOF dataset to {OUT_STAGE1_PATH}")
    agg.to_parquet(OUT_STAGE1_PATH, index=False)

    # Summary diagnostics (train-only)
    pos_nogender = agg[(agg["y_init"] == 1) & (agg["has_gender"] == 0)]
    pos_gender = agg[(agg["y_init"] == 1) & (agg["has_gender"] == 1)]
    neg_alpha = agg[(agg["y_init"] == 0) & (agg["is_alpha_space"] == 1)]

    report = {
        "raw_train_rows": raw_train_rows,
        "val_overlap_rows_removed": overlap_rows,
        "val_overlap_unique_removed": overlap_unique,
        "unique_clean_rows": unique_rows,
        "conflicting_texts_resolved": conflict_count,
        "y_init_pos": int((agg["y_init"] == 1).sum()),
        "y_init_neg": int((agg["y_init"] == 0).sum()),
        "pos_with_gender": len(pos_gender),
        "pos_without_gender": len(pos_nogender),
        "neg_pure_alpha": len(neg_alpha),
        "pos_nogender_low_oof_lt_0_25": int((pos_nogender["oof_tfidf"] < 0.25).sum()),
        "pos_nogender_low_oof_lt_0_50": int((pos_nogender["oof_tfidf"] < 0.50).sum()),
        "pos_gender_low_oof_lt_0_25": int((pos_gender["oof_tfidf"] < 0.25).sum()),
        "pos_with_single_neg_token": int(((agg["y_init"] == 1) & (agg["oof_tok_any_single_neg"] == 1)).sum()),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(OUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("--> Stage 1 Report:")
    print(json.dumps(report, indent=2))
    print(f"    Sample suspicious y_init=1 (oof_tfidf < 0.15):")
    susp = agg[(agg["y_init"] == 1) & (agg["oof_tfidf"] < 0.15)].sort_values("oof_tfidf")
    print(susp[["text_norm", "has_gender", "oof_tfidf", "oof_tok_any_single_neg"]].head(20).to_string())


if __name__ == "__main__":
    main()
