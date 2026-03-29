from pathlib import Path

import streamlit as st


st.set_page_config(page_title="IR Query GUI", page_icon=":mag:", layout="wide")

st.title("Information Retrieval Query GUI")
st.caption("Boolean and proximity retrieval on indexed speech documents")

st.markdown(
    """
### Quick Guide
1. Create a virtual environment and install requirements.
2. Run `python preprocessing.py` once to build indexes.
3. Use this page to run queries and inspect matched documents.

### Supported Query Formats
- Boolean: `biggest and ( near or box )`
- Proximity: `after year /1`

Proximity interpretation for `term1 term2 /k`:
- `term2` appears exactly `k` positions after `term1`.
"""
)

INDEX_FILE = Path("indexes") / "index_data.pkl"
DATASET_DIR = Path("dataset")

if not INDEX_FILE.exists():
    st.error("Index file not found. Please run `python preprocessing.py` first.")
    st.stop()

# Import only after index existence check, because main.py exits when file is missing.
import main as retrieval_engine  # noqa: E402


def load_document_text(document_id):
    file_path = DATASET_DIR / f"speech_{document_id}.txt"
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


query = st.text_input("Enter query", placeholder="Example: biggest and ( near or box )")

if st.button("Search", type="primary"):
    cleaned_query = query.strip()
    if not cleaned_query:
        st.warning("Please enter a query.")
        st.stop()

    preprocessed_query = retrieval_engine.preprocessQuery(cleaned_query)

    st.write("Preprocessed query tokens:", preprocessed_query)

    if not retrieval_engine.isValidQuery(preprocessed_query):
        st.error("Invalid query.")
        st.stop()

    document_ids = retrieval_engine.getRelevantDocumentIDs(preprocessed_query)
    sorted_ids = sorted(document_ids)

    st.success(f"Found {len(sorted_ids)} matching document(s).")
    st.write("Document IDs:", sorted_ids)

    for document_id in sorted_ids:
        with st.expander(f"Document {document_id}"):
            document_text = load_document_text(document_id)
            if document_text is None:
                st.error("Document file not found.")
            else:
                st.text_area(
                    label=f"Content of speech_{document_id}.txt",
                    value=document_text,
                    height=280,
                    key=f"doc_{document_id}",
                )
