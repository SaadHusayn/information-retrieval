import linecache
import pickle
import re
from pathlib import Path

from nltk.stem.porter import PorterStemmer
from sortedcontainers import SortedDict, SortedList


SPEECH_CONTENT_LINE_NO = 2
STOPWORDS_FILE = "stopwords.txt"
DATASET_DIR = Path("dataset")
INDEX_DIR = Path("indexes")
INDEX_FILE = INDEX_DIR / "index_data.pkl"


def load_stopwords(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file]


def documentProcessing(document_content, stopwords, porter_stemmer):
    # truecasing
    document_content = document_content.lower()

    # removing text in square brackets that are no speech eg [Applause]
    document_content = re.sub(r"\[.*?\]", "", document_content)

    # removing punctuations: replacing .:,- with (space) and removing '?
    trans_table = str.maketrans(".:,-—–", "      ", "'\"?$0123456789()")
    tokens = document_content.translate(trans_table).split()

    # removing stop words
    tokens = [token for token in tokens if token not in stopwords]

    # stemming using porter stemmer
    tokens = [porter_stemmer.stem(token) for token in tokens]

    return tokens


def addTokensToInvertedIndex(inverted_index, document_tokens, document_id):
    for token in document_tokens:
        if inverted_index.get(token) is None:
            inverted_index[token] = SortedList([document_id])
        else:
            inverted_index[token].add(document_id)


def addTokensToPositionalIndex(positional_index, document_tokens, document_id):
    for index, token in enumerate(document_tokens):
        if positional_index.get(token) is None:
            positional_index[token] = SortedDict({document_id: SortedList([index])})
        elif positional_index[token].get(document_id) is None:
            positional_index[token][document_id] = SortedList([index])
        else:
            positional_index[token][document_id].add(index)


def sorted_document_ids(dataset_dir):
    ids = []
    for file_path in dataset_dir.glob("speech_*.txt"):
        stem = file_path.stem
        try:
            ids.append(int(stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


def preProccessingPipeline():
    porter_stemmer = PorterStemmer()
    stopwords = load_stopwords(STOPWORDS_FILE)

    inverted_index = SortedDict()
    positional_index = SortedDict()

    document_ids = sorted_document_ids(DATASET_DIR)

    for document_id in document_ids:
        document_path = DATASET_DIR / f"speech_{document_id}.txt"
        document_content = linecache.getline(str(document_path), SPEECH_CONTENT_LINE_NO)
        if not document_content:
            continue

        document_tokens = documentProcessing(document_content, stopwords, porter_stemmer)

        # removing duplicates for inverted index
        addTokensToInvertedIndex(inverted_index, list(set(document_tokens)), document_id)
        addTokensToPositionalIndex(positional_index, document_tokens, document_id)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "wb") as file:
        pickle.dump(
            {
                "num_documents": len(document_ids),
                "all_document_ids": set(document_ids),
                "inverted_index": inverted_index,
                "positional_index": positional_index,
            },
            file,
        )

    print(f"Saved indexes to {INDEX_FILE}")
    print(f"Indexed documents: {len(document_ids)}")


if __name__ == "__main__":
    preProccessingPipeline()
