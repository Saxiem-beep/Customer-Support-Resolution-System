# Customer Support Resolution Engine

A support agent pastes a new customer message. The app finds the most similar tickets the team has already resolved, then drafts a reply grounded only in those resolutions, with citations like `[TKT-002]` so the agent can verify every claim. Nothing is sent automatically: a human owns what goes out.

It runs in two modes and degrades gracefully between them:

| Mode | Retrieval | Draft |
|---|---|---|
| Full RAG (`OPENAI_API_KEY` set) | OpenAI `text-embedding-3-small` + in-memory ChromaDB (cosine space) | `gpt-4o-mini`, temperature 0.2, max 500 tokens |
| Base Mode (no key, fully offline) | TF-IDF (English stop words, 1–2 grams, sublinear TF) | Deterministic template |

If any part of the full path fails, the app drops one rung down the ladder and **says so on screen**: the UI always shows the exact pipeline used (for example `TF-IDF + template fallback`) plus a warning with the reason.

## Architecture

```
app.py            Streamlit UI. Only calls engine.run_pipeline().
engine.py
  Data layer       load_tickets()        load, validate and clean the CSV
  Retrieval layer  TfidfRetriever        both expose search(query, top_k)
                   ChromaRetriever       and return the same dict shape
  Generation layer generate_with_openai  LLM draft
                   template_reply        deterministic fallback
  Orchestrator     run_pipeline()        picks the best path, handles fallbacks,
                                         checks citations
eval_retrieval.py Retrieval smoke test (Recall@1, Recall@5, MRR)
tests/            57 tests, fake OpenAI client, headless UI via AppTest
```

Because both retrievers share one interface, falling back is one line, and swapping in BM25, hybrid search or pgvector is one new class with no UI changes.

## Setup (Windows, VS Code terminal)

```powershell
git clone https://github.com/Saxiem-beep/Customer-Support-Resolution-System.git
cd Customer-Support-Resolution-System
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux: activate with `source venv/bin/activate` instead.

Optional, for Full RAG mode (never commit the key):

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

## Run

```powershell
streamlit run app.py
```

Open the URL it prints (usually `http://localhost:8501`).

## Test

```powershell
pytest -q
python eval_retrieval.py
```

The tests never call the real API. A fake OpenAI client returns deterministic embeddings and canned replies, so the full semantic path runs against a real ChromaDB instance, along with the failure paths: chat failing, chat returning empty, embeddings failing at build and mid-query, the model citing a ticket that doesn't exist, six kinds of broken CSV, and an API key leaking into an error message. GitHub Actions runs both commands on every push.

## How it handles hallucination and prompt injection

Reduced and contained by design, not solved:

- **Grounding.** The system prompt makes the retrieved tickets the only evidence and forbids inventing policies, amounts or timelines. Similarity scores are passed to the model, and a low top score adds an explicit "say you need to check" note.
- **Citation post-check.** Every `[ID]` in the draft is extracted and compared against the retrieved set. Invented IDs show as a red error. The evidence panel updates live as the agent edits the draft.
- **Untrusted input.** The customer message is wrapped in `<customer_message>` tags, and `<`, `>`, `&` are escaped in it and in every ticket field, so a customer can't type `</customer_message>` and close the tag early.
- **Injection flag.** A pattern check catches obvious attempts ("ignore all previous instructions") and forces the "let me check" reply. It is easy to bypass; see Known limitations.
- **No authority.** The model can only write text, and a human approves it. If it ever got tools (issuing refunds), authorization would have to be enforced server-side, outside the model.
- **Secrets.** The key is read only from `OPENAI_API_KEY` and never displayed. Errors go through `safe_error`, which masks the key and anything shaped like `sk-...`. The cache key uses a 12-character SHA-256 fingerprint, never the key.

## Failure handling

| Failure | What the app does |
|---|---|
| `openai` / `chromadb` not installed or fails at import | Import wrapped in `try/except Exception` (chromadb can raise `RuntimeError` on an old SQLite, not just `ImportError`), Base Mode |
| No `OPENAI_API_KEY` | TF-IDF + template, sidebar shows "Not set" |
| Embedding or ChromaDB fails at index build | Warning, failure remembered in `session_state` so reruns don't retry the API, TF-IDF |
| Semantic search fails mid-query | Warning, same query re-run on TF-IDF |
| Chat call fails, times out (30 s, 2 retries) or returns empty | Warning, deterministic template |
| CSV missing / empty / unparseable / wrong columns / no usable rows | Clear error with the fix, app stops cleanly with no traceback |
| Duplicate IDs, missing IDs, missing category, missing resolution, extra columns | Cleaned (keep first duplicate, `AUTO-0003`-style IDs, `Uncategorized`, drop the row, drop the column), warnings in the sidebar |
| Empty, symbols-only or over-4,000-character query | Friendly warning, no search |

## Retrieval results (TF-IDF, 18 tickets, `python eval_retrieval.py`)

Each intent is tested with a direct phrasing and a paraphrase:

| | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| Direct phrasing (5 queries) | 1.00 | 1.00 | 1.00 |
| Paraphrase (5 queries) | 0.40 | 0.80 | 0.60 |

Direct top scores ranged from 0.26 to 0.66. Paraphrases are where TF-IDF breaks: "You billed me double for one purchase" shares no words with "charged twice" and scores 0.00. That gap is the reason the semantic mode exists. The prompt-injection test message scores 0.23, above the 0.15 threshold (it shares "refund" with real refund tickets), so the threshold alone does not catch it; the injection flag routes it to a human. This is a 10-query smoke test, not a benchmark.

## Known limitations

- **Evaluation is a smoke test.** Thresholds (0.15 TF-IDF, 0.30 semantic) are heuristics on a handful of queries. A real eval needs a few hundred labelled queries. The semantic mode and reply quality have not been measured against the real OpenAI API; the mocks prove control flow and fallbacks only.
- **Thresholds are fragile.** Extra words dilute TF-IDF scores: "I was charged twice and want my money back" scores 0.147, just under 0.15. Paraphrases can also be confidently wrong (one scored 0.17 on the wrong ticket).
- **The injection flag is a speed bump.** Paraphrased or non-English injections get past it. The real safeguards are escaping, grounding and the human in the loop.
- **Data.** A CSV is not a database, ticket policies can go stale, and there is no PII redaction before text is sent to the API.
- **Scale.** Every ticket is embedded at startup into an in-memory store. Fine for 18 tickets, not for 2 million: that needs an offline indexing job, a persistent vector store, and incremental updates keyed by content hash.
- **Not production-ready.** No auth, monitoring, rate limiting or audit log. Production-minded, not production-ready.

## Configuration

All optional, set as environment variables: `TICKETS_PATH`, `CHAT_MODEL`, `EMBED_MODEL`, `TOP_K` (5), `TFIDF_THRESHOLD` (0.15), `SEMANTIC_THRESHOLD` (0.30), `TEMPERATURE` (0.2), `MAX_TOKENS` (500), `OPENAI_TIMEOUT` (30), `OPENAI_MAX_RETRIES` (2), `MAX_QUERY_CHARS` (4000).
