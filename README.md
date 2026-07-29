# Customer Support Resolution Engine

A retrieval-augmented generation (RAG) tool that finds similar past support tickets and drafts citation-backed response suggestions for new customer queries. Built with a dual-backend architecture so it works with or without an OpenAI API key.

## Features

- **Semantic ticket retrieval** — finds the most relevant resolved tickets for a new customer query
- **Dual retrieval backends**
  - `ChromaDB` + OpenAI `text-embedding-3-small` embeddings (production mode)
  - `TF-IDF` + cosine similarity (offline fallback, no API key required)
- **Automatic fallback** — if the OpenAI path fails or no key is set, the app transparently degrades to TF-IDF retrieval
- **Citation-backed drafts** — generated responses cite the specific ticket IDs used as context, via `gpt-4.1-mini`
- **Interactive UI** — built with Streamlit for quick demos and manual testing

## How it works

1. Historical tickets are loaded from a CSV and flattened into a single text blob per ticket (ID, category, issue, customer message, resolution)
2. On a new query, the app embeds/searches for the top-k most similar tickets
3. Retrieved tickets are passed as context to an LLM, which drafts a support reply citing the relevant ticket IDs
4. Results (retrieved tickets + draft) are displayed in the UI for review/editing

## Project structure

```
.
├── app.py                 # Main Streamlit application
├── sample_tickets.csv      # Sample historical ticket dataset
└── chroma_db/              # Persisted ChromaDB collection (created on first run)
```

## Requirements

- Python 3.9+
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `chromadb`
- `openai`

Install with:

```bash
pip install streamlit pandas numpy scikit-learn chromadb openai
```

## Setup

1. Clone the repo and place your ticket data at `sample_tickets.csv` with at least these columns:
   - `ticket_id`
   - `category`
   - `issue`
   - `customer_message`
   - `resolution`

2. (Optional, for full RAG mode) Set your OpenAI API key:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

   Without a key, the app runs entirely offline using TF-IDF retrieval and a templated draft response.

## Running the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## Usage

1. Enter a customer query in the text area
2. Choose how many similar tickets to retrieve (k)
3. Click **Generate draft**
4. Review the retrieved tickets and the generated draft response, and edit as needed before sending

## Retrieval modes

| Mode | Trigger | Method |
|---|---|---|
| Full RAG | `OPENAI_API_KEY` set and reachable | ChromaDB + OpenAI embeddings, GPT-generated draft |
| Fallback | API call fails | TF-IDF cosine similarity, GPT-generated draft |
| Offline | No `OPENAI_API_KEY` | TF-IDF cosine similarity, templated draft |

## Notes

- The Chroma collection is built once and persisted to `./chroma_db`; delete this folder to force re-indexing after updating `sample_tickets.csv`.
- This project is a lightweight, single-file companion to a fuller LlamaIndex/ChromaDB notebook implementation of the same system.

## License

MIT (or update as appropriate for your use case).
