import re
import pickle
import sys
from pathlib import Path

from nltk.stem.porter import PorterStemmer


INDEX_FILE = Path("indexes") / "index_data.pkl"
porter_stemmer = PorterStemmer()


def loadIndexes(index_file_path):
    if not index_file_path.exists():
        print(f"Index file not found: {index_file_path}")
        print("Run preprocessing.py first to build the indexes.")
        sys.exit(1)

    with open(index_file_path, "rb") as file:
        saved_data = pickle.load(file)

    return (
        saved_data["inverted_index"],
        saved_data["positional_index"],
        saved_data["all_document_ids"],
    )


inverted_index, positional_index, ALL_DOCUMENT_IDS_SET = loadIndexes(INDEX_FILE)

def preprocessQuery(query):
    # Ensure brackets are split as standalone tokens: "(term" -> "(" "term"
    query = re.sub(r'([()])', r' \1 ', query)
    query = query.strip().lower().split()

    preprocessed_tokens = []
    for token in query:
        # Support compact positional syntax like /3 by splitting it to "/" and "3".
        if len(token) > 1 and token[0] == '/' and token[1:].isdecimal():
            preprocessed_tokens.append('/')
            preprocessed_tokens.append(token[1:])
            continue

        # Keep operators/special tokens unchanged, stem only actual terms.
        if token in {"and", "or", "not", "(", ")", "/"} or token.isdecimal():
            preprocessed_tokens.append(token)
        else:
            preprocessed_tokens.append(porter_stemmer.stem(token))

    return preprocessed_tokens

def isValidQuery(preprocessed_query):
    if len(preprocessed_query) == 0:
        return False

    #positional query
    if len(preprocessed_query) == 4 and preprocessed_query[2] == '/' and preprocessed_query[3].isdecimal():
        return True
    else:
        #normal boolean query
        if preprocessed_query[0] == "and" or preprocessed_query[0] == "or" or preprocessed_query[-1] == "and" or preprocessed_query[-1] == "or" or preprocessed_query[-1] == "not":
            return False
        
        return True

def isPositionalQuery(preprocessed_query):
    return len(preprocessed_query) == 4 and preprocessed_query[2] == '/' and preprocessed_query[3].isdecimal()

def getRelevantDocumentIDs(preprocessed_query):
    result_set = set()
    if isPositionalQuery(preprocessed_query):
        k = int(preprocessed_query[3])
        term1 = preprocessed_query[0]
        term2 = preprocessed_query[1]

        if positional_index.get(term1) is not None and positional_index.get(term2) is not None:
            common_docs = set(positional_index[term1].keys()) & set(positional_index[term2].keys())

            for document_id in common_docs:
                term1_positions = positional_index[term1][document_id]
                term2_positions = positional_index[term2][document_id]


                # Two-pointer scan on sorted position lists.
                i = 0
                j = 0
                while i < len(term1_positions) and j < len(term2_positions):
                    pos1 = term1_positions[i]
                    pos2 = term2_positions[j]
                    distance = pos2 - pos1

                    # Ordered proximity: term2 appears exactly k positions after term1.
                    if distance == k:
                        result_set.add(document_id)
                        break

                    if pos2 <= pos1:
                        j += 1
                    elif distance > k:
                        i += 1
                    else:
                        j += 1
    elif '(' in preprocessed_query or ')' in preprocessed_query:
        # Recursive descent parsing for bracketed boolean queries.
        # Grammar:
        # expression := term ("or" term)*
        # term       := factor ("and" factor)*
        # factor     := "not" factor | "(" expression ")" | WORD
        idx = 0

        def docs_for_token(token):
            if inverted_index.get(token) is None:
                return set()
            return set(inverted_index[token])

        def parse_expression():
            nonlocal idx
            current_result = parse_term()

            while idx < len(preprocessed_query) and preprocessed_query[idx] == 'or':
                idx += 1
                rhs = parse_term()
                current_result = current_result | rhs

            return current_result

        def parse_term():
            nonlocal idx
            current_result = parse_factor()

            while idx < len(preprocessed_query) and preprocessed_query[idx] == 'and':
                idx += 1
                rhs = parse_factor()
                current_result = current_result & rhs

            return current_result

        def parse_factor():
            nonlocal idx
            if idx >= len(preprocessed_query):
                return set()

            token = preprocessed_query[idx]

            if token == 'not':
                idx += 1
                return ALL_DOCUMENT_IDS_SET - parse_factor()

            if token == '(':
                idx += 1
                inside_result = parse_expression()
                if idx < len(preprocessed_query) and preprocessed_query[idx] == ')':
                    idx += 1
                return inside_result

            if token in {'and', 'or', ')'}:
                return set()

            idx += 1
            return docs_for_token(token)

        result_set = parse_expression()

    else:
        #normal boolean query
        
        idx = 0
        perform_not = False
        perform_and = False
        while idx < len(preprocessed_query):
            term = preprocessed_query[idx]
            if term == 'not':
                perform_not = not perform_not
            elif term == 'and':
                perform_and = True
            elif term == 'or':
                perform_and = False
            else:
                #if a word appears
                if inverted_index.get(term) is not None:
                    term_documents = set(inverted_index[term])
                    if perform_not:
                        term_documents = ALL_DOCUMENT_IDS_SET - term_documents
                        perform_not = False
                    
                    if perform_and:
                        result_set = result_set & term_documents
                        perform_and = False
                    else:
                        result_set = result_set | term_documents
                    
            idx += 1
    return result_set

if __name__ == "__main__":
    query = input()
    preprocessed_query = preprocessQuery(query)
    if isValidQuery(preprocessed_query):
        document_ids = getRelevantDocumentIDs(preprocessed_query)
        print(document_ids)
    else:
        print("Invalid query")