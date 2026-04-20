import linecache
import pickle
import re
import math
from pathlib import Path
from collections import defaultdict

from nltk.stem import WordNetLemmatizer


SPEECH_CONTENT_LINE_NO = 2
STOPWORDS_FILE = "Stopword-List.txt"
DATASET_DIR = Path("Trump Speechs")
INDEX_DIR = Path("indexes")
INDEX_FILE = INDEX_DIR / "index_data.pkl"


def load_stopwords(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return set(line.strip() for line in file if line.strip())


def documentProcessing(document_content, stopwords, lemmatizer):
    # case folding
    document_content = document_content.lower()

    # removing text in square brackets eg [Applause]
    document_content = re.sub(r"\[.*?\]", "", document_content)

    # removing punctuation: replacing .:,- with space and removing '?$digits()
    trans_table = str.maketrans(".:,-—–", "      ", "'\"?$0123456789()")
    tokens = document_content.translate(trans_table).split()

    # removing stop words
    tokens = [token for token in tokens if token not in stopwords]

    # lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def sorted_document_ids(dataset_dir):
    ids = []
    for file_path in dataset_dir.glob("speech_*.txt"):
        stem = file_path.stem
        try:
            ids.append(int(stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


def preProcessingPipeline():
    lemmatizer = WordNetLemmatizer()
    stopwords = load_stopwords(STOPWORDS_FILE)

    document_ids = sorted_document_ids(DATASET_DIR)
    num_documents = len(document_ids)

    # tf[doc_id][term] = raw term frequency
    tf = {}
    # df[term] = number of documents containing the term
    df = defaultdict(int)

    for document_id in document_ids:
        document_path = DATASET_DIR / f"speech_{document_id}.txt"
        document_content = linecache.getline(str(document_path), SPEECH_CONTENT_LINE_NO)
        if not document_content:
            continue

        tokens = documentProcessing(document_content, stopwords, lemmatizer)

        # compute term frequency for this document
        term_freq = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1

        tf[document_id] = dict(term_freq)

        # update document frequency
        for term in term_freq:
            df[term] += 1

    # compute idf = log((N + 1) / (df_t + 1)) with smoothing
    idf = {}
    for term, doc_freq in df.items():
        idf[term] = math.log((num_documents + 1) / (doc_freq + 1))

    # save index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "wb") as file:
        pickle.dump(
            {
                "num_documents": num_documents,
                "document_ids": document_ids,
                "tf": tf,
                "idf": idf,
            },
            file,
        )

    print(f"Saved index to {INDEX_FILE}")
    print(f"Indexed documents: {num_documents}")
    print(f"Vocabulary size: {len(idf)}")


if __name__ == "__main__":
    preProcessingPipeline()
    # print(sorted_document_ids(DATASET_DIR))
