"""
Accuracy Evaluation Module
Conversational Recommendation System

Metrics:
- Top-1 Accuracy: Is the top recommended category in expected list?
- Top-3 Accuracy: Is any of top-3 in expected list?
- Precision@K, Recall@K
- Sentiment Accuracy: Does polarity match expected sentiment label?
"""

import json
import sys
import os

# Allow running standalone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import recommend, sentiment_analysis, PRODUCT_DATA

TEST_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "test_cases.json")


def evaluate():
    with open(TEST_PATH) as f:
        test_cases = json.load(f)

    total = len(test_cases)
    top1_hits = 0
    top3_hits = 0
    sentiment_hits = 0
    precision_sum = 0.0
    recall_sum = 0.0

    results = []

    print("\n" + "=" * 70)
    print(f"{'CONVERSATIONAL RECOMMENDATION SYSTEM — ACCURACY EVALUATION':^70}")
    print("=" * 70)
    print(f"{'#':<4} {'Input Sentence':<45} {'Top1':>5} {'Top3':>5} {'Sent':>5}")
    print("-" * 70)

    for i, case in enumerate(test_cases):
        sentence = case["sentence"]
        expected = set(case["expected"])
        expected_sentiment = case["sentiment"]

        result = recommend(sentence, PRODUCT_DATA, top_n=3)
        recs = [r["category"] for r in result["recommendations"]]

        # ── Top-1 accuracy ────────────────────────────────────────────────────
        top1_hit = len(recs) > 0 and recs[0] in expected
        top1_hits += int(top1_hit)

        # ── Top-3 accuracy ────────────────────────────────────────────────────
        top3_hit = any(r in expected for r in recs)
        top3_hits += int(top3_hit)

        # ── Precision & Recall @3 ─────────────────────────────────────────────
        hits_at_3 = len(set(recs) & expected)
        precision_k = hits_at_3 / len(recs) if recs else 0
        recall_k = hits_at_3 / len(expected) if expected else 0
        precision_sum += precision_k
        recall_sum += recall_k

        # ── Sentiment accuracy ────────────────────────────────────────────────
        predicted_sent = "positive" if result["is_positive"] else "negative"
        sent_hit = predicted_sent == expected_sentiment
        sentiment_hits += int(sent_hit)

        results.append({
            "sentence": sentence,
            "expected": list(expected),
            "predicted": recs,
            "top1": top1_hit,
            "top3": top3_hit,
            "precision@3": round(precision_k, 3),
            "recall@3": round(recall_k, 3),
            "polarity": result["polarity"],
            "sentiment_correct": sent_hit
        })

        t1 = "✓" if top1_hit else "✗"
        t3 = "✓" if top3_hit else "✗"
        s  = "✓" if sent_hit  else "✗"
        label = sentence[:44]
        print(f"{i+1:<4} {label:<45} {t1:>5} {t3:>5} {s:>5}")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    top1_acc    = top1_hits / total * 100
    top3_acc    = top3_hits / total * 100
    sent_acc    = sentiment_hits / total * 100
    avg_prec    = precision_sum / total * 100
    avg_recall  = recall_sum / total * 100
    f1          = (2 * avg_prec * avg_recall / (avg_prec + avg_recall)
                   if (avg_prec + avg_recall) > 0 else 0)

    print("=" * 70)
    print(f"\n{'RESULTS SUMMARY':^70}")
    print("-" * 70)
    print(f"  Total test cases        : {total}")
    print(f"  Top-1 Accuracy          : {top1_hits}/{total}  ({top1_acc:.1f}%)")
    print(f"  Top-3 Accuracy          : {top3_hits}/{total}  ({top3_acc:.1f}%)")
    print(f"  Sentiment Accuracy      : {sentiment_hits}/{total}  ({sent_acc:.1f}%)")
    print(f"  Avg Precision@3         : {avg_prec:.1f}%")
    print(f"  Avg Recall@3            : {avg_recall:.1f}%")
    print(f"  F1-Score                : {f1:.1f}%")
    print("=" * 70)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "..", "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "total": total,
                "top1_accuracy": round(top1_acc, 2),
                "top3_accuracy": round(top3_acc, 2),
                "sentiment_accuracy": round(sent_acc, 2),
                "avg_precision_at_3": round(avg_prec, 2),
                "avg_recall_at_3": round(avg_recall, 2),
                "f1_score": round(f1, 2)
            },
            "per_case": results
        }, f, indent=2)

    print(f"\n  Full results saved to: evaluation_results.json\n")
    return results


if __name__ == "__main__":
    evaluate()
