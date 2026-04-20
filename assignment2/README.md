# Assignment 2 — Vector Space Model (VSM) Information Retrieval

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
cd assignment2

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install nltk streamlit
```

### 3. Download NLTK Data (WordNet)

```bash
python3 -c "import nltk; nltk.download('wordnet')"
```

### 4. Build the Index

```bash
python3 preprocessing.py
```

This reads all 56 speeches from `Trump Speechs/`, preprocesses them (case folding, stopword removal, lemmatization), computes TF and IDF values, and saves the index to `indexes/index_data.pkl`.

### 5. Run the CLI

```bash
python3 main.py
```

Type a query and press Enter. Type `quit` to exit.

### 6. Run the GUI

```bash
streamlit run app.py
```

This opens a Streamlit web app in your browser with a search interface, adjustable alpha threshold, and ranked results with speech titles and snippets.
