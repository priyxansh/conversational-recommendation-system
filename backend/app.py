"""
Conversational Recommendation System
Based on: "Conversational Recommendation System Using NLP and Sentiment Analysis"
Talegaonkar et al., IJAEN, Volume-12, Issue-6, June 2024

Backend: Flask REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import operator
import os
import logging

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── NLTK Downloads ───────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

app = Flask(__name__)
CORS(app)

# ─── Load GloVe Word Vectors ──────────────────────────────────────────────────
logger.info("Loading GloVe word vectors (glove-wiki-gigaword-50)...")
import gensim.downloader as gensim_api
word_vectors = gensim_api.load("glove-wiki-gigaword-50")
logger.info("Word vectors loaded successfully.")

# ─── Load Product Dataset ─────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "products.json")
with open(DATASET_PATH, "r") as f:
    PRODUCT_DATA = json.load(f)

logger.info(f"Loaded {len(PRODUCT_DATA)} product categories: {list(PRODUCT_DATA.keys())}")

# ─── Important POS Tags (only nouns, adjectives, verbs) ───────────────────────
IMP_POS_TAGS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}

stop_words = set(stopwords.words("english"))

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_word_vector(word: str):
    """Return the GloVe vector for a word, or None if OOV."""
    try:
        return word_vectors[word.lower()]
    except KeyError:
        return None


def cosine_sim(v1, v2) -> float:
    """Cosine similarity between two 1-D numpy vectors."""
    v1 = np.array(v1).reshape(1, -1)
    v2 = np.array(v2).reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def tokenize_and_filter(text: str):
    """
    Tokenize the input text, remove stop-words, and keep only
    words whose POS tag is in IMP_POS_TAGS.
    Returns (filtered_words, pos_tags_list)
    """
    tokens = word_tokenize(text.lower())
    # Remove stop-words and non-alphabetic tokens
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    # POS tagging
    pos_tags = pos_tag(filtered)
    important = [word for word, tag in pos_tags if tag in IMP_POS_TAGS]
    return important, pos_tags


def sentiment_analysis(text: str):
    """
    Use TextBlob to determine polarity.
    Threshold: 0.2 (as per paper) — below this is treated as negative intent.
    Returns (polarity_score, is_positive)
    """
    polarity = TextBlob(text).sentiment.polarity
    is_positive = polarity >= 0.2
    return polarity, is_positive


def recommend(text: str, product_data: dict, top_n: int = 3):
    """
    Full recommendation pipeline:
    1. Tokenize & filter
    2. Sentiment analysis
    3. Stage 1: Compare words with category headers
    4. Stage 2: Compare words with keywords (weighted by frequency)
    Returns dict with recommendations and metadata.
    """
    # Step 1 – Tokenize
    imp_words, pos_tags = tokenize_and_filter(text)
    logger.info(f"Important words: {imp_words}")

    # Step 2 – Sentiment
    polarity, is_positive = sentiment_analysis(text)
    logger.info(f"Polarity: {polarity:.3f}  |  Positive intent: {is_positive}")

    if not imp_words:
        return {
            "recommendations": [],
            "important_words": [],
            "polarity": polarity,
            "is_positive": is_positive,
            "error": "No meaningful words found in input."
        }

    keys = list(product_data.keys())
    category_scores = {k: 0.0 for k in keys}

    for word in imp_words:
        word_vec = get_word_vector(word)
        if word_vec is None:
            continue

        # ── Stage 1: Header comparison ────────────────────────────────────────
        header_scores = {}
        for key in keys:
            key_vec = get_word_vector(key)
            if key_vec is not None:
                header_scores[key] = cosine_sim(word_vec, key_vec)

        # Top 3 headers for this word
        top3_headers = sorted(header_scores, key=header_scores.get,
                              reverse=is_positive)[:3]

        # ── Stage 2: Keyword comparison (weighted average) ────────────────────
        for category in top3_headers:
            keywords = product_data[category]
            total_weight = 0
            weighted_sum = 0.0
            for kw, freq in keywords.items():
                kw_vec = get_word_vector(kw)
                if kw_vec is not None:
                    sim = cosine_sim(word_vec, kw_vec)
                    weighted_sum += freq * sim
                    total_weight += freq
            if total_weight > 0:
                category_scores[category] += weighted_sum / total_weight

    # Sort categories by cumulative score
    sorted_categories = sorted(category_scores.items(),
                                key=operator.itemgetter(1),
                                reverse=is_positive)

    top_categories = sorted_categories[:top_n]

    recommendations = [
        {
            "category": cat,
            "score": round(score, 4),
            "keywords": list(product_data[cat].keys())[:8]
        }
        for cat, score in top_categories if score > 0
    ]

    return {
        "recommendations": recommendations,
        "important_words": imp_words,
        "polarity": round(polarity, 4),
        "is_positive": is_positive,
        "all_scores": {k: round(v, 4) for k, v in sorted_categories}
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "categories": list(PRODUCT_DATA.keys())})


@app.route("/recommend", methods=["POST"])
def recommend_route():
    """
    POST /recommend
    Body: { "text": "I need a new dress", "product_data": {...} (optional) }
    Returns top-3 product recommendations.
    """
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Allow client to send personalised product_data (for adaptive learning)
    product_data = body.get("product_data", PRODUCT_DATA)

    result = recommend(text, product_data)
    logger.info(f"Query: '{text}'  →  {[r['category'] for r in result['recommendations']]}")
    return jsonify(result)


@app.route("/categories", methods=["GET"])
def categories():
    """Return all product categories with their top keywords."""
    return jsonify({
        cat: list(kw.keys())[:5] for cat, kw in PRODUCT_DATA.items()
    })


@app.route("/sentiment", methods=["POST"])
def sentiment_route():
    """Standalone sentiment analysis endpoint."""
    body = request.get_json(silent=True) or {}
    text = body.get("text", "")
    polarity, is_positive = sentiment_analysis(text)
    subjectivity = TextBlob(text).sentiment.subjectivity
    return jsonify({
        "polarity": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "is_positive": is_positive,
        "label": "positive" if is_positive else "negative"
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
