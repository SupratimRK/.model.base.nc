import os
import sys
import json
import time
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CompactNameClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_config.json")
TOKENIZER_DIR = os.path.join(MODELS_DIR, "tokenizer")
ONNX_PATH = os.path.join(MODELS_DIR, "model_int8.onnx")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "model_weights.pt")

class NameClassifier:
    """
    Dedicated binary Name vs Non-Name predictor.
    Supports ultra-fast ONNX INT8 CPU inference or PyTorch.
    """
    def __init__(self, use_onnx=True):
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}. Please train the model first.")

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.max_length = self.cfg.get("max_length", 20)
        self.labels = {int(k): v for k, v in self.cfg.get("labels", {"0": "NON_NAME", "1": "REAL_NAME"}).items()}

        # Load Tokenizer
        tok_json = os.path.join(TOKENIZER_DIR, "tokenizer.json")
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_json,
            do_lower_case=True,
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]"
        )

        self.use_onnx = use_onnx and os.path.exists(ONNX_PATH)
        if self.use_onnx:
            import onnxruntime
            session_opts = onnxruntime.SessionOptions()
            session_opts.intra_op_num_threads = 1
            session_opts.inter_op_num_threads = 1
            session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = onnxruntime.InferenceSession(ONNX_PATH, session_opts)
            self.backend_name = "ONNX Runtime INT8 (Single-Thread)"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = CompactNameClassifier(
                vocab_size=self.cfg["vocab_size"],
                emb_dim=self.cfg["emb_dim"],
                hidden_dim=self.cfg["hidden_dim"],
                num_layers=self.cfg["num_layers"],
                num_heads=self.cfg["num_heads"],
                max_len=self.max_length,
                dropout=0.0
            ).to(self.device)
            self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True))
            self.model.eval()
            self.backend_name = f"PyTorch ({self.device})"

    def softmax(self, logits):
        e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="np"
        )

        input_ids = enc["input_ids"].astype(np.int32)
        attention_mask = enc["attention_mask"].astype(np.int32)

        if self.use_onnx:
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            logits = self.session.run(None, ort_inputs)[0]
            probs = self.softmax(logits)
        else:
            with torch.no_grad():
                ids_t = torch.tensor(input_ids, device=self.device)
                mask_t = torch.tensor(attention_mask, device=self.device)
                logits_t = self.model(ids_t, mask_t)
                probs = torch.softmax(logits_t, dim=-1).cpu().numpy()

        results = []
        for i, text in enumerate(texts):
            pred_idx = int(np.argmax(probs[i]))
            label = self.labels[pred_idx]
            conf = float(probs[i][pred_idx])

            results.append({
                "text": text,
                "classification": label,
                "confidence": conf,
                "raw_probs": {
                    "non_name": float(probs[i][0]),
                    "real_name": float(probs[i][1])
                }
            })
        return results

def run_test_suite(classifier):
    test_cases = [
        "Supratim Mondal",
        "Sayantika Biswas",
        "Ankush Dutta",
        "Priya Sharma",
        "John Smith",
        "Mar",
        "Kumar",
        "get",
        "mar",
        "get mar",
        "0fxfps",
        "ShadowHunter99",
        "112_SAYANTIKA BISWAS",
        "AYAN CSE3048",
        "toyexpo",
        "fluffynsint",
        "cecipfp"
    ]

    print(f"\n--- Running Evaluation using [{classifier.backend_name}] ---")
    results = classifier.predict(test_cases)
    col_w = [26, 14, 12, 36]
    header = f"{'Input Text':<{col_w[0]}} | {'Classification':<{col_w[1]}} | {'Confidence':<{col_w[2]}} | {'Raw Probabilities':<{col_w[3]}}"
    print(header)
    print("-" * len(header))
    for res in results:
        txt = (res['text'][:23] + '...') if len(res['text']) > 25 else res['text']
        p = res['raw_probs']
        raw_str = f"Real: {p['real_name']*100:5.1f}% | Non: {p['non_name']*100:5.1f}%"
        print(f"{txt:<{col_w[0]}} | {res['classification']:<{col_w[1]}} | {res['confidence']*100:>9.1f}% | {raw_str:<{col_w[3]}}")

def main():
    use_pytorch = "--pytorch" in sys.argv
    show_raw = "--raw" in sys.argv
    clean_argv = [a for a in sys.argv if a not in ("--pytorch", "--onnx", "--raw")]

    classifier = NameClassifier(use_onnx=(not use_pytorch))

    if len(clean_argv) > 1 and clean_argv[1] == "--interactive":
        print(f"\n[{classifier.backend_name}] Enter strings to classify (type 'exit' or 'quit' to stop):")
        while True:
            try:
                line = input("\nQuery > ").strip()
                if not line or line.lower() in ("exit", "quit"):
                    break
                t0 = time.perf_counter()
                res = classifier.predict(line)[0]
                dt_ms = (time.perf_counter() - t0) * 1000
                p = res['raw_probs']
                print(f"  Result:     {res['classification']} ({res['confidence']*100:.1f}%) [took {dt_ms:.2f} ms]")
                print(f"  Raw Probs:  Real: {p['real_name']*100:.2f}% | Non-Name: {p['non_name']*100:.2f}%")
            except (KeyboardInterrupt, EOFError):
                break
    elif len(clean_argv) > 1:
        queries = clean_argv[1:]
        results = classifier.predict(queries)
        for res in results:
            p = res['raw_probs']
            raw_str = f" | [Real: {p['real_name']*100:.1f}%, Non: {p['non_name']*100:.1f}%]" if show_raw else ""
            print(f"'{res['text']}' => {res['classification']} ({res['confidence']*100:.1f}%){raw_str}")
    else:
        run_test_suite(classifier)

if __name__ == "__main__":
    main()
