# Conversational Recommendation System

A natural language-driven product recommendation engine designed to improve Customer Relationship Management (CRM) and consumer intent tracking. By utilizing advanced Natural Language Processing (NLP) and Sentiment Analysis, this system allows users to express their needs conversationally rather than relying on exact keyword searches.

This project is based on the research paper: *"Conversational Recommendation System Using NLP and Sentiment Analysis"* (Talegaonkar et al., IJAEN, Volume-12, Issue-6, June 2024).

## 🎯 How it Works (Marketing Management Perspective)

In digital marketing and e-commerce, **Personalization** and **Consumer Intent** are critical. This backend API minimizes friction in the sales funnel by functioning as a "smart shopkeeper":
1. **Sentiment Analysis:** Detects the user's mood (positive/negative) using TextBlob. This prevents recommending products a user explicitly says they dislike.
2. **Semantic Matching:** Instead of rigid keyword matching, it uses **GloVe Word Vectors** to understand synonyms and related concepts conceptually (e.g., matching "jogging" to the "footwear" and "sports" categories).
3. **Adaptive Learning Hook:** The API naturally supports accepting tailored product weightings from the frontend, allowing for integration with long-term user profiles and adaptive CRM databases.

---

## 🏗️ Project Structure

```text
.
├── backend/
│   ├── app.py                 # The core Flask REST API server
│   ├── evaluate.py            # Accuracy evaluation script (uses full NLP models)
│   └── requirements.txt       # Python dependencies
├── dataset/
│   ├── products.json          # 10 categories with frequency-weighted keywords
│   └── test_cases.json        # 25 conversational test cases and expected outputs
├── evaluate_standalone.py     # Lightweight evaluation script (zero dependencies)
└── README.md                  # This file
```

---

## 📦 Requirements

- **Python**: 3.10, 3.11, or 3.12 recommended (Python 3.13 requires `uv` to handle building legacy dependencies properly).
- **Network Access**: The server requires internet access on the very first run to download NLTK corpora and the GloVe word vector dataset (~66 MB).

---

## 🚀 Setup & Installation

Below are the setup instructions varying slightly by operating system. Because some dependencies (`scikit-learn==1.4.2`, `numpy==1.26.4`) are pinned, modern versions of Python may struggle to build them without downgrading `scipy`. Applying `scipy<1.13` is strongly recommended across all platforms to ensure compatibility with `gensim==4.3.2`.

### 🐧 Linux (Arch Linux via `uv`)
Arch Linux rolls very fast (usually Python 3.13+). Using the `uv` package manager is the most robust way to ensure a stable environment.

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment specifically using Python 3.11 (downloads automatically if needed):
   ```bash
   uv venv --python 3.11
   ```
3. Activate the virtual environment:
   ```bash
   # Bash/Zsh
   source .venv/bin/activate
   
   # Fish shell
   source .venv/bin/activate.fish
   ```
4. Install the requirements and patch `scipy`:
   ```bash
   uv pip install -r requirements.txt
   uv pip install "scipy<1.13"
   ```
5. Run the server:
   ```bash
   python app.py
   ```

### 🍎 macOS
1. Open your terminal and navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   pip install "scipy<1.13"
   ```
4. Run the server:
   ```bash
   python app.py
   ```

### 🪟 Windows
1. Open Command Prompt or PowerShell and navigate to the backend directory:
   ```cmd
   cd backend
   ```
2. Create and activate a virtual environment:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies using pip:
   ```cmd
   pip install -r requirements.txt
   pip install "scipy<1.13"
   ```
4. Run the server:
   ```cmd
   python app.py
   ```

---

## 🧪 Testing the API

Once the server indicates it is running (`* Running on http://127.0.0.1:5000`), you can test the recommendation endpoint:

**cURL Example:**
```bash
curl -X POST http://127.0.0.1:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"text": "I need running shoes for jogging"}'
```

**Expected JSON Response snippet:**
```json
{
  "important_words": ["need", "running", "shoes", "jogging"],
  "is_positive": true,
  "recommendations": [
    {
      "category": "footwear",
      "score": 0.5927
    },
    {
      "category": "sports",
      "score": 0.3652
    }
  ]
}
```

## 📊 Running Evaluations
We provide two distinct evaluation scripts to gauge the accuracy of the system against our `dataset/test_cases.json`.

1. **Full Evaluation (`backend/evaluate.py`)**: 
   Must be run from the backend directory with the virtual environment activated. It uses the full GloVe vector implementation to measure Top-1, Top-3, Sentiment Accuracy, and F1-score.
   ```bash
   cd backend
   python evaluate.py
   ```

2. **Standalone Simulator (`evaluate_standalone.py`)**:
   Runs without *any* external libraries and uses basic character tri-grams to simulate logic. Useful for quick demonstrations without downloading GloVe models.
   ```bash
   python evaluate_standalone.py
   ```
