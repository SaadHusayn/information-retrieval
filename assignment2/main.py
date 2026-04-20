import re
import pickle
import sys
import math
from pathlib import Path
from collections import defaultdict

from nltk.stem import WordNetLemmatizer


INDEX_FILE = Path("indexes") / "index_data.pkl"
STOPWORDS_FILE = "Stopword-List.txt"
ALPHA = 0.005

lemmatizer = WordNetLemmatizer()


def load_stopwords(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return set(line.strip() for line in file if line.strip())


stopwords = load_stopwords(STOPWORDS_FILE)


def loadIndex(index_file_path):
    if not index_file_path.exists():
        print(f"Index file not found: {index_file_path}")
        print("Run preprocessing.py first to build the index.")
        sys.exit(1)

    with open(index_file_path, "rb") as file:
        saved_data = pickle.load(file)

    return (
        saved_data["num_documents"],
        saved_data["document_ids"],
        saved_data["tf"],
        saved_data["idf"],
    )


num_documents, document_ids, tf, idf = loadIndex(INDEX_FILE)


def preprocessQuery(query):
    """Preprocess query: case fold, remove punctuation, remove stopwords, lemmatize."""
    query = query.strip().lower()

    # remove punctuation same way as documents
    trans_table = str.maketrans(".:,-—–", "      ", "'\"?$0123456789()")
    tokens = query.translate(trans_table).split()

    # remove stopwords
    tokens = [token for token in tokens if token not in stopwords]

    # lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def computeDocumentTFIDF(doc_id):
    """Compute tf-idf vector for a document. Returns dict: term -> tf*idf."""
    doc_tf = tf.get(doc_id, {})
    tfidf_vector = {}
    for term, freq in doc_tf.items():
        if term in idf:
            tfidf_vector[term] = freq * idf[term]
    return tfidf_vector


def computeQueryTFIDF(query_tokens):
    """Compute tf-idf vector for the query. Returns dict: term -> tf*idf."""
    query_tf = defaultdict(int)
    for token in query_tokens:
        query_tf[token] += 1

    tfidf_vector = {}
    for term, freq in query_tf.items():
        if term in idf:
            tfidf_vector[term] = freq * idf[term]
    return tfidf_vector


def cosineSimilarity(vec_a, vec_b):
    """Compute cosine similarity between two sparse vectors (dicts)."""
    # dot product: only terms in both vectors contribute
    dot_product = 0.0
    for term in vec_a:
        if term in vec_b:
            dot_product += vec_a[term] * vec_b[term]

    if dot_product == 0.0:
        return 0.0

    # magnitudes
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot_product / (mag_a * mag_b)


def searchVSM(query_text):
    """Search using VSM and return sorted results above alpha threshold."""
    query_tokens = preprocessQuery(query_text)

    if not query_tokens:
        return []

    query_vector = computeQueryTFIDF(query_tokens)

    if not query_vector:
        return []

    results = []
    for doc_id in document_ids:
        doc_vector = computeDocumentTFIDF(doc_id)
        similarity = cosineSimilarity(query_vector, doc_vector)
        if similarity >= ALPHA:
            results.append((doc_id, similarity))

    # sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def formatResults(query_text, results):
    """Format results in the style of Query List VSM.txt."""
    doc_id_set = {str(doc_id) for doc_id, _ in results}
    print(f"\nquery='{query_text}'")
    print(f"\nLength={len(results)}")
    print(doc_id_set)


def main():
    print("=" * 60)
    print("  Vector Space Model (VSM) Information Retrieval System")
    print("=" * 60)
    print(f"Documents indexed: {num_documents}")
    print(f"Vocabulary size: {len(idf)}")
    print(f"Alpha threshold: {ALPHA}")
    print("-" * 60)

    while True:
        query = input("\nEnter query (or 'quit' to exit): ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        results = searchVSM(query)
        formatResults(query, results)


if __name__ == "__main__":
    main()
