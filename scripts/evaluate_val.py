import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import NameClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
VAL_PATH = os.path.join(PROJECT_ROOT, "curated_datasets", "val_curated.parquet")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_config.json")
OUT_METRICS_PATH = os.path.join(PROJECT_ROOT, "curated_datasets", "val_eval_report.json")


def evaluate_mode(clf: NameClassifier, df_va: pd.DataFrame, consolidate_words: bool = False, word_threshold: float = None):
    texts = df_va["text"].fillna("").astype(str).tolist()
    y_true = df_va["is_name"].to_numpy()
    preds = np.zeros(len(texts), dtype=np.int64)
    probs = np.zeros(len(texts), dtype=np.float32)

    batch_size = 8192
    for i in range(0, len(texts), batch_size):
        res = clf.predict(
            texts[i:i + batch_size],
            consolidate_words=consolidate_words,
            word_threshold=word_threshold,
        )
        for j, r in enumerate(res):
            preds[i + j] = 1 if r["classification"] == "REAL_NAME" else 0
            probs[i + j] = r["raw_probs"]["real_name"]

    acc = float(accuracy_score(y_true, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary")
    cm = confusion_matrix(y_true, preds, labels=[0, 1])

    by_status = {}
    for status, idx in df_va.groupby("audit_status").groups.items():
        idx_arr = np.fromiter(idx, dtype=np.int64)
        s_true = y_true[idx_arr]
        s_pred = preds[idx_arr]
        s_acc = float(accuracy_score(s_true, s_pred))
        s_cm = confusion_matrix(s_true, s_pred, labels=[0, 1])
        by_status[status] = {
            "count": int(len(idx_arr)),
            "accuracy": round(s_acc, 4),
            "fp": int(s_cm[0, 1]),
            "fn": int(s_cm[1, 0]),
        }

    return {
        "accuracy": round(acc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "by_audit_status": by_status,
    }


def main():
    t0 = time.time()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    clf = NameClassifier(use_onnx=True)
    df_va = pd.read_parquet(VAL_PATH)

    print("===============================================================")
    print(f"Evaluating Frozen Validation Split ({len(df_va):,} rows)")
    print(f"Backend: {clf.backend_name} | Temp={clf.temperature:.4f} | Threshold={clf.threshold:.4f}")
    print("===============================================================")

    single_pass = evaluate_mode(clf, df_va, consolidate_words=False)
    print(
        f"\n[Single-Pass Calibrated] Acc: {single_pass['accuracy']*100:.2f}% | "
        f"Prec: {single_pass['precision']*100:.2f}% | Rec: {single_pass['recall']*100:.2f}% | "
        f"F1: {single_pass['f1']*100:.2f}%"
    )
    for status, m in single_pass["by_audit_status"].items():
        print(f"  {status:26s} | n={m['count']:7,} | acc={m['accuracy']*100:6.2f}% | FP={m['fp']:6,} | FN={m['fn']:6,}")

    consolidated = evaluate_mode(clf, df_va, consolidate_words=True, word_threshold=0.10)
    print(
        f"\n[Word-Consolidated (w_tau=0.10)] Acc: {consolidated['accuracy']*100:.2f}% | "
        f"Prec: {consolidated['precision']*100:.2f}% | Rec: {consolidated['recall']*100:.2f}% | "
        f"F1: {consolidated['f1']*100:.2f}%"
    )
    for status, m in consolidated["by_audit_status"].items():
        print(f"  {status:26s} | n={m['count']:7,} | acc={m['accuracy']*100:6.2f}% | FP={m['fp']:6,} | FN={m['fn']:6,}")

    report = {
        "model_config": cfg,
        "val_rows": len(df_va),
        "single_pass_calibrated": single_pass,
        "word_consolidated_w010": consolidated,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(OUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n--> Saved evaluation report to {OUT_METRICS_PATH}")


if __name__ == "__main__":
    main()
