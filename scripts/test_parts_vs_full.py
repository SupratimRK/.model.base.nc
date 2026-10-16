import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import NameClassifier

def main():
    clf = NameClassifier(use_onnx=True)

    test_cases = [
        # Legitimate Clean Names
        "Supratim Mondal",
        "Sayantika Biswas",
        "John Smith",
        "Priya Sharma",
        "Sayantika Rani Biswas",
        "Aarav Kumar Gupta",
        "Supratim",
        "Mondal",
        "Mar",
        # Contaminated Names (Real name + Noise/Roll/Dept)
        "112_SAYANTIKA BISWAS",
        "AYAN CSE3048",
        "Supratim Mondal 2024",
        "Supratim_dev",
        # Pure Non-Names / Usernames
        "ShadowHunter99",
        "0fxfps",
        "toyexpo",
        "fluffynsint",
        # Dictionary / Ambiguous Combos
        "get",
        "mar",
        "get mar",
        "call john",
        "test user"
    ]

    print("==========================================================================================================")
    print(f"{'Input String':<24} | {'Full-String Inference':<22} | {'Part-by-Part (Consolidated)':<28} | {'Part Breakdown'}")
    print("==========================================================================================================")

    for s in test_cases:
        # 1. Full string
        full_res = clf.predict(s)[0]
        full_label = full_res['classification']
        full_conf = full_res['confidence'] * 100
        full_str = f"{full_label} ({full_conf:.1f}%)"

        # 2. Part by part
        parts = s.split()
        part_results = clf.predict(parts)

        # Consolidation Strategy:
        # A name is REAL if:
        # - More than 50% of words are REAL_NAME and none have strong noise (< 10% real)
        # OR: Strict AND (all parts must be REAL_NAME)
        strict_and = all(p['classification'] == 'REAL_NAME' for p in part_results)
        min_real = min(p['raw_probs']['real_name'] for p in part_results) * 100
        avg_real = (sum(p['raw_probs']['real_name'] for p in part_results) / len(part_results)) * 100

        consolidated_label = "REAL_NAME" if strict_and else "NON_NAME"
        consolidated_str = f"{consolidated_label} (min: {min_real:.1f}%)"

        breakdown_items = []
        for p in part_results:
            tag = "REAL" if p['classification'] == 'REAL_NAME' else "NON"
            breakdown_items.append(f"{p['text']}={tag}({p['confidence']*100:.0f}%)")
        breakdown_str = ", ".join(breakdown_items)

        print(f"{s:<24} | {full_str:<22} | {consolidated_str:<28} | {breakdown_str}")

    print("==========================================================================================================")

if __name__ == "__main__":
    main()
