"""
Conversational Recommendation System — Standalone Accuracy Evaluator
No external libraries required. Uses the same logic as the full backend.
Run: python evaluate_standalone.py
"""

import json, math, re, os

# ─── Product Dataset ──────────────────────────────────────────────────────────
PRODUCT_DATA = {
  "clothing":    {"dress":10,"shirt":9,"jeans":9,"jacket":8,"skirt":8,"blouse":7,"suit":7,"coat":6,"sweater":6,"hoodie":6,"kurta":8,"saree":8,"wear":5,"outfit":6,"fabric":5,"fashion":6,"clothes":9,"garment":4,"wardrobe":4,"attire":4},
  "electronics": {"phone":10,"laptop":10,"tablet":8,"headphones":8,"charger":7,"camera":8,"television":8,"speaker":7,"keyboard":6,"mouse":6,"monitor":7,"smartwatch":7,"earbuds":7,"router":5,"gadget":5,"device":6,"tech":5,"wireless":6,"bluetooth":6,"digital":5,"smart":6,"battery":5,"screen":5},
  "footwear":    {"shoes":10,"sneakers":9,"boots":8,"sandals":8,"heels":7,"slippers":7,"loafers":6,"running":6,"sports":5,"formal":5,"casual":5,"leather":5,"footwear":8,"walk":4,"feet":4,"comfort":5,"ankle":4},
  "furniture":   {"sofa":10,"chair":9,"table":9,"bed":9,"desk":8,"wardrobe":7,"shelf":7,"cabinet":7,"couch":8,"mattress":8,"drawer":6,"bookcase":6,"lamp":5,"rug":5,"curtain":5,"wooden":5,"interior":5,"decor":6,"home":5,"room":5,"living":5,"bedroom":6,"cushion":5,"pillow":5},
  "groceries":   {"food":10,"vegetables":9,"fruits":9,"milk":8,"bread":8,"rice":8,"flour":7,"oil":7,"sugar":7,"salt":6,"spices":6,"eggs":7,"cheese":6,"yogurt":6,"butter":6,"organic":5,"fresh":6,"snacks":6,"cereal":5,"juice":6,"grocery":8,"cook":5,"meal":5},
  "books":       {"book":10,"novel":9,"read":8,"author":7,"fiction":8,"nonfiction":7,"textbook":8,"literature":7,"story":6,"poetry":6,"biography":6,"science":6,"history":6,"thriller":6,"mystery":6,"fantasy":6,"romance":6,"education":6,"study":6,"knowledge":5,"chapter":5,"library":5},
  "sports":      {"cricket":10,"football":9,"gym":9,"fitness":9,"yoga":8,"cycling":7,"swimming":7,"tennis":7,"basketball":7,"badminton":7,"exercise":8,"workout":8,"training":7,"equipment":6,"ball":6,"racket":5,"sport":8,"athletic":5,"run":6,"strength":5,"cardio":6,"outdoor":5},
  "beauty":      {"lipstick":9,"skincare":10,"makeup":10,"perfume":8,"shampoo":8,"moisturizer":8,"foundation":7,"eyeliner":7,"blush":6,"serum":7,"conditioner":7,"facewash":7,"sunscreen":7,"cream":6,"lotion":6,"hair":7,"skin":7,"beauty":8,"glow":5,"fragrance":6,"cosmetics":7,"grooming":5},
  "toys":        {"toy":10,"game":8,"puzzle":8,"lego":7,"doll":7,"kids":8,"children":8,"baby":7,"play":6,"fun":5,"creative":5,"blocks":5,"stuffed":5,"cartoon":4,"coloring":4,"educational":6},
  "kitchen":     {"cookware":10,"pan":9,"pot":9,"knife":8,"blender":8,"microwave":8,"mixer":7,"spatula":6,"strainer":5,"grater":5,"utensil":7,"appliance":6,"cook":7,"bake":6,"kitchen":8,"oven":7,"refrigerator":8}
}

STOP_WORDS = {
  "i","a","an","the","is","are","was","were","be","been","being","have","has","had",
  "do","does","did","will","would","could","should","may","might","shall","can",
  "to","of","in","for","on","with","at","by","from","up","about","into","through",
  "during","before","after","above","below","my","your","his","her","its","our",
  "their","this","that","these","those","it","we","they","you","he","she","and",
  "or","but","not","no","nor","so","yet","both","either","neither","just","very",
  "also","here","there","where","when","how","what","who","which","some","any",
  "all","each","few","more","most","other","own","same","than","then","too",
  "need","want","looking","give","me","new","good","nice","best","get","buy",
  "find","like","use"
}

NEG_WORDS = ["don't","dont","not","never","no","without","hate","avoid","dislike","stop","terrible","worst","bad","horrible","refuse","deny"]
POS_WORDS = ["love","like","want","need","great","good","best","nice","wonderful","beautiful","excellent","amazing","buy","get"]

# ─── Trigram-based cosine similarity (no GloVe needed) ───────────────────────
def ngrams(word, n=3):
    word = word.lower()
    return set(word[i:i+n] for i in range(max(1, len(word)-n+1)))

def cosine_sim(w1, w2):
    a, b = ngrams(w1), ngrams(w2)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / math.sqrt(len(a) * len(b))

# ─── Tokenizer ────────────────────────────────────────────────────────────────
def tokenize(text):
    words = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]

# ─── Sentiment analysis ───────────────────────────────────────────────────────
def get_sentiment(text):
    lower = text.lower()
    score = 0.0
    for w in NEG_WORDS:
        if w in lower: score -= 0.4
    for w in POS_WORDS:
        if w in lower: score += 0.2
    score = max(-1.0, min(1.0, score))
    return round(score, 4), score >= 0.2

# ─── Recommendation engine ────────────────────────────────────────────────────
def recommend(text, product_data=PRODUCT_DATA, top_n=3):
    words = tokenize(text)
    polarity, is_positive = get_sentiment(text)

    if not words:
        return {"recommendations": [], "important_words": [], "polarity": polarity, "is_positive": is_positive}

    categories = list(product_data.keys())
    scores = {c: 0.0 for c in categories}

    for word in words:
        # Stage 1: header similarity
        header_sims = {c: cosine_sim(word, c) for c in categories}
        top3_headers = sorted(categories, key=lambda c: header_sims[c], reverse=is_positive)[:3]

        # Stage 2: keyword weighted average
        for cat in top3_headers:
            kws = product_data[cat]
            weighted_sum, total_weight = 0.0, 0
            for kw, freq in kws.items():
                sim = cosine_sim(word, kw)
                weighted_sum += freq * sim
                total_weight += freq
            if total_weight > 0:
                scores[cat] += weighted_sum / total_weight

    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=is_positive)
    top3 = [{"category": c, "score": round(s, 4), "keywords": list(product_data[c].keys())[:6]}
            for c, s in sorted_cats[:top_n]]

    return {"recommendations": top3, "important_words": words, "polarity": polarity, "is_positive": is_positive}

# ─── Test Cases ───────────────────────────────────────────────────────────────
TEST_CASES = [
  {"sentence": "I need a new dress for the party",                        "expected": ["clothing","beauty"],      "sentiment": "positive"},
  {"sentence": "I want to buy some fresh vegetables and fruits",          "expected": ["groceries"],              "sentiment": "positive"},
  {"sentence": "Looking for a good laptop for college",                   "expected": ["electronics","books"],    "sentiment": "positive"},
  {"sentence": "I need running shoes for jogging",                        "expected": ["footwear","sports"],      "sentiment": "positive"},
  {"sentence": "Can you suggest a good novel to read",                    "expected": ["books"],                  "sentiment": "positive"},
  {"sentence": "I want wireless headphones with good bass",               "expected": ["electronics"],            "sentiment": "positive"},
  {"sentence": "Need a comfortable sofa for my living room",              "expected": ["furniture"],              "sentiment": "positive"},
  {"sentence": "Looking for gym equipment and yoga mat",                  "expected": ["sports"],                 "sentiment": "positive"},
  {"sentence": "I want to buy lipstick and moisturizer",                  "expected": ["beauty"],                 "sentiment": "positive"},
  {"sentence": "My kid needs educational toys and puzzles",               "expected": ["toys","books"],           "sentiment": "positive"},
  {"sentence": "I need a good pressure cooker and blender",               "expected": ["kitchen"],                "sentiment": "positive"},
  {"sentence": "I dont want electronics give me something for my bedroom","expected": ["furniture"],              "sentiment": "negative"},
  {"sentence": "I need some cricket gear and football",                   "expected": ["sports"],                 "sentiment": "positive"},
  {"sentence": "Looking for a formal suit and leather shoes",             "expected": ["clothing","footwear"],    "sentiment": "positive"},
  {"sentence": "Want organic food and cooking spices",                    "expected": ["groceries","kitchen"],    "sentiment": "positive"},
  {"sentence": "I need a study table and bookcase for my room",           "expected": ["furniture","books"],      "sentiment": "positive"},
  {"sentence": "Need shampoo and skincare products",                      "expected": ["beauty"],                 "sentiment": "positive"},
  {"sentence": "Looking for board games for my children",                 "expected": ["toys"],                   "sentiment": "positive"},
  {"sentence": "I dont need clothes I want some gadgets",                 "expected": ["electronics"],            "sentiment": "negative"},
  {"sentence": "I want a smartphone and a smartwatch",                    "expected": ["electronics"],            "sentiment": "positive"},
  {"sentence": "Need a new saree for the festival",                       "expected": ["clothing"],               "sentiment": "positive"},
  {"sentence": "I want to buy rice flour and cooking oil",                "expected": ["groceries","kitchen"],    "sentiment": "positive"},
  {"sentence": "Looking for sandals and comfortable footwear",            "expected": ["footwear"],               "sentiment": "positive"},
  {"sentence": "I need a science textbook for my studies",                "expected": ["books"],                  "sentiment": "positive"},
  {"sentence": "Want a camera and some accessories for photography",      "expected": ["electronics"],            "sentiment": "positive"},
]

# ─── Evaluate ─────────────────────────────────────────────────────────────────
def evaluate():
    total = len(TEST_CASES)
    top1_hits = top3_hits = sent_hits = 0
    prec_sum = rec_sum = 0.0
    all_results = []

    print("\n" + "═"*80)
    print(f"{'CONVERSATIONAL RECOMMENDATION SYSTEM — ACCURACY EVALUATION':^80}")
    print(f"{'Talegaonkar et al., IJAEN 2024  |  Standalone Simulation':^80}")
    print("═"*80)
    print(f"{'#':<4}  {'Input Sentence':<47}  {'T1':>3}  {'T3':>3}  {'S':>3}  {'Predicted'}")
    print("─"*80)

    for i, tc in enumerate(TEST_CASES):
        sentence  = tc["sentence"]
        expected  = set(tc["expected"])
        exp_sent  = tc["sentiment"]

        res  = recommend(sentence)
        recs = [r["category"] for r in res["recommendations"]]

        t1 = recs[0] in expected if recs else False
        t3 = any(r in expected for r in recs)
        top1_hits += int(t1); top3_hits += int(t3)

        hits = len(set(recs) & expected)
        p = hits / len(recs) if recs else 0
        r_ = hits / len(expected) if expected else 0
        prec_sum += p; rec_sum += r_

        pred_sent = "positive" if res["is_positive"] else "negative"
        s_ok = pred_sent == exp_sent
        sent_hits += int(s_ok)

        all_results.append({
            "id": i+1, "sentence": sentence, "expected": list(expected),
            "predicted_top3": recs, "top1": t1, "top3": t3,
            "precision@3": round(p,3), "recall@3": round(r_,3),
            "polarity": res["polarity"], "sentiment_ok": s_ok
        })

        t1s = "✓" if t1 else "✗"; t3s = "✓" if t3 else "✗"; ss = "✓" if s_ok else "✗"
        print(f"{i+1:<4}  {sentence[:47]:<47}  {t1s:>3}  {t3s:>3}  {ss:>3}  {', '.join(recs)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    t1a = top1_hits/total*100; t3a = top3_hits/total*100; sa = sent_hits/total*100
    avgp = prec_sum/total*100; avgr = rec_sum/total*100
    f1 = 2*avgp*avgr/(avgp+avgr) if (avgp+avgr) > 0 else 0

    print("═"*80)
    print(f"\n{'RESULTS SUMMARY':^80}")
    print("─"*80)
    print(f"  Total test cases           : {total}")
    print(f"  Top-1 Accuracy             : {top1_hits}/{total}  ({t1a:.1f}%)")
    print(f"  Top-3 Accuracy             : {top3_hits}/{total}  ({t3a:.1f}%)")
    print(f"  Sentiment Accuracy         : {sent_hits}/{total}  ({sa:.1f}%)")
    print(f"  Avg Precision@3            : {avgp:.1f}%")
    print(f"  Avg Recall@3               : {avgr:.1f}%")
    print(f"  F1-Score                   : {f1:.1f}%")
    print("─"*80)
    print(f"  Note: Similarity via trigram n-grams (standalone mode).")
    print(f"  Full accuracy uses GloVe vectors — run backend/app.py with pip deps.")
    print("═"*80)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")
    summary = {"total":total,"top1_accuracy":round(t1a,2),"top3_accuracy":round(t3a,2),
               "sentiment_accuracy":round(sa,2),"avg_precision_at_3":round(avgp,2),
               "avg_recall_at_3":round(avgr,2),"f1_score":round(f1,2)}
    with open(out_path,"w") as f:
        json.dump({"summary": summary, "per_case": all_results}, f, indent=2)
    print(f"\n  Results written → {out_path}\n")
    return summary

if __name__ == "__main__":
    evaluate()
