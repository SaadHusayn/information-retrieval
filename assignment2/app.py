import streamlit as st
import pickle
import re
import math
import linecache
from pathlib import Path
from collections import defaultdict

from nltk.stem import WordNetLemmatizer


# ----- Configuration -----

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "indexes" / "index_data.pkl"
STOPWORDS_FILE = BASE_DIR / "Stopword-List.txt"
DATASET_DIR = BASE_DIR / "Trump Speechs"
ALPHA_DEFAULT = 0.005


# ----- Loading Functions -----

@st.cache_resource
def load_stopwords(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


@st.cache_resource
def load_index(index_file_path):
    if not index_file_path.exists():
        return None
    with open(index_file_path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def get_lemmatizer():
    return WordNetLemmatizer()


def get_speech_title(doc_id):
    """Get the title (line 1) of a speech file."""
    path = DATASET_DIR / f"speech_{doc_id}.txt"
    title = linecache.getline(str(path), 1).strip()
    return title if title else f"Speech {doc_id}"


def get_speech_snippet(doc_id, max_chars=300):
    """Get a short snippet (line 2) of a speech file."""
    path = DATASET_DIR / f"speech_{doc_id}.txt"
    content = linecache.getline(str(path), 2).strip()
    if len(content) > max_chars:
        return content[:max_chars] + "..."
    return content


# ----- VSM Functions -----

def preprocess_query(query, stopwords, lemmatizer):
    """Preprocess query: case fold, remove punctuation, remove stopwords, lemmatize."""
    query = query.strip().lower()
    trans_table = str.maketrans(".:,-\u2014\u2013", "      ", "'\"?$0123456789()")
    tokens = query.translate(trans_table).split()
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def compute_document_tfidf(doc_id, tf, idf):
    """Compute tf-idf vector for a document."""
    doc_tf = tf.get(doc_id, {})
    return {term: freq * idf[term] for term, freq in doc_tf.items() if term in idf}


def compute_query_tfidf(query_tokens, idf):
    """Compute tf-idf vector for the query."""
    query_tf = defaultdict(int)
    for token in query_tokens:
        query_tf[token] += 1
    return {term: freq * idf[term] for term, freq in query_tf.items() if term in idf}


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two sparse vectors (dicts)."""
    dot_product = sum(vec_a[t] * vec_b[t] for t in vec_a if t in vec_b)
    if dot_product == 0.0:
        return 0.0
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_product / (mag_a * mag_b)


def search_vsm(query_text, stopwords, lemmatizer, index_data, alpha):
    """Search using VSM and return sorted results above alpha threshold."""
    query_tokens = preprocess_query(query_text, stopwords, lemmatizer)
    if not query_tokens:
        return [], query_tokens

    tf = index_data["tf"]
    idf = index_data["idf"]
    document_ids = index_data["document_ids"]

    query_vector = compute_query_tfidf(query_tokens, idf)
    if not query_vector:
        return [], query_tokens

    results = []
    for doc_id in document_ids:
        doc_vector = compute_document_tfidf(doc_id, tf, idf)
        similarity = cosine_similarity(query_vector, doc_vector)
        if similarity >= alpha:
            results.append((doc_id, similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return results, query_tokens


# ----- Streamlit App -----

def main():
    st.set_page_config(
        page_title="VSM Search Engine",
        page_icon=":mag:",
        layout="wide",
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-title {
        text-align: center;
        padding: 0.5rem 0;
    }
    .result-card {
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-radius: 0 8px 8px 0;
    }
    .result-card-title {
        color: #1a73e8;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .result-card-score {
        color: #5f6368;
        font-size: 0.85rem;
    }
    .result-card-snippet {
        color: #3c4043;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .stat-box {
        background-color: #e8f0fe;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #5f6368;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("<h1 class='main-title'>🔍 Vector Space Model Search Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #5f6368;'>Trump Speeches Collection (2015-2016) &mdash; CS4051 Information Retrieval</p>", unsafe_allow_html=True)
    st.divider()

    # Load resources
    stopwords = load_stopwords(STOPWORDS_FILE)
    lemmatizer = get_lemmatizer()
    index_data = load_index(INDEX_FILE)

    if index_data is None:
        st.error("Index not found. Please run `python preprocessing.py` first to build the index.")
        return

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        alpha = st.slider(
            "Alpha Threshold (α)",
            min_value=0.0,
            max_value=0.1,
            value=ALPHA_DEFAULT,
            step=0.001,
            format="%.4f",
            help="Minimum cosine similarity score for a document to be included in results.",
        )

        st.divider()
        st.header("📊 Index Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-number'>{index_data['num_documents']}</div>
                <div class='stat-label'>Documents</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-number'>{len(index_data['idf']):,}</div>
                <div class='stat-label'>Vocabulary</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.header("📝 Sample Queries")
        sample_queries = [
            "massive inflow of refugees",
            "pakistan afghanistan",
            "Hillary Clinton",
            "American Energy Revolution",
            "peaceful change",
            "muslims",
            "Global interests",
        ]
        for sq in sample_queries:
            if st.button(sq, key=f"sample_{sq}", use_container_width=True):
                st.session_state["query_input"] = sq

    # Search bar
    query = st.text_input(
        "Enter your search query:",
        value=st.session_state.get("query_input", ""),
        placeholder="Type a query and press Enter...",
        key="query_box",
    )

    # Sync session state
    if query != st.session_state.get("query_input", ""):
        st.session_state["query_input"] = query

    if query.strip():
        results, query_tokens = search_vsm(query, stopwords, lemmatizer, index_data, alpha)

        # Query info
        st.markdown(f"**Preprocessed tokens:** `{' | '.join(query_tokens)}`")

        if not results:
            st.warning("No documents matched your query above the alpha threshold.")
        else:
            # Results summary
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Documents Found", len(results))
            with col_b:
                st.metric("Highest Score", f"{results[0][1]:.6f}")
            with col_c:
                st.metric("Lowest Score", f"{results[-1][1]:.6f}")

            st.divider()

            # Document ID set (matching gold standard format)
            doc_id_set = {str(doc_id) for doc_id, _ in results}
            with st.expander(f"📋 Document ID Set (Length={len(results)})", expanded=False):
                st.code(str(doc_id_set))

            # Results list
            st.subheader(f"Search Results ({len(results)} documents)")
            for rank, (doc_id, score) in enumerate(results, 1):
                title = get_speech_title(doc_id)
                snippet = get_speech_snippet(doc_id)
                st.markdown(f"""
                <div class='result-card'>
                    <div class='result-card-title'>#{rank} &mdash; Speech {doc_id}: {title}</div>
                    <div class='result-card-score'>Cosine Similarity: {score:.6f}</div>
                    <div class='result-card-snippet'>{snippet}</div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
