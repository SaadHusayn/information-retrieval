# Information Retrieval Assignment 1

This project builds an inverted index and positional index for the speech dataset, then answers boolean and proximity queries.

## Project Files

- `preprocessing.py`: Reads dataset files, preprocesses text, builds indexes, and saves them to disk.
- `main.py`: Loads saved indexes and runs query-time retrieval.
- `requirements.txt`: Python dependencies.
- `stopwords.txt`: Stopword list used during preprocessing.

## Setup Guide

### 1. Create a virtual environment

From the project root:

```bash
python3 -m venv .venv
```

### 2. Activate the virtual environment

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Build Indexes

Run preprocessing once (or whenever dataset/stopwords change):

```bash
python preprocessing.py
```

This creates:

- `indexes/index_data.pkl`

## Run Query Engine

```bash
python main.py
```

Then type a query and press Enter.

## Query Examples

Boolean query:

```text
biggest and ( near or box )
```

Proximity query (exact ordered distance):

```text
after year /1
```

Interpretation of proximity query `term1 term2 /k`:
- `term2` must appear exactly `k` positions after `term1` in a document.
