import os
import re
import streamlit as st
import engine

DATA_PATH = os.getenv("TICKETS_PATH", "sample_tickets.csv")
st.set_page_config(page_title="Support Resolution Engine", layout="wide")
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~])")


def safe_md(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _MD_SPECIAL.sub(r"\\\1", text)


def file_stamp(path):
    try:
        s = os.stat(path)
        return s.st_mtime, s.st_size
    except OSError:
        return 0.0, 0


@st.cache_data(show_spinner=False)
def cached_load(path, mtime, size):
    return engine.load_tickets(path)


@st.cache_resource(show_spinner="Building TF-IDF index...")
def cached_tfidf(signature, _df):
    return engine.TfidfRetriever(_df)


@st.cache_resource(show_spinner=False)
def cached_client(fingerprint, _api_key):
    return engine.make_openai_client(_api_key)


@st.cache_resource(show_spinner="Embedding tickets with OpenAI...")
def cached_semantic(signature, model, fingerprint, _df, _client):
    return engine.ChromaRetriever(_df, _client, model)


def failure_key(signature, fingerprint):
    return f"semantic_failed::{signature[:12]}::{engine.EMBED_MODEL}::{fingerprint}"


st.title("Customer Support Resolution Engine")
st.caption("Finds similar resolved tickets and drafts a reply grounded only in them, with citations the agent can verify. Nothing is sent automatically.")

try:
    df, data_warnings = cached_load(DATA_PATH, *file_stamp(DATA_PATH))
except engine.DataError as exc:
    st.error(f"Can't load tickets: {exc}")
    st.stop()

signature = engine.dataset_signature(df)
tfidf = cached_tfidf(signature, df)
api_key = os.getenv("OPENAI_API_KEY") or None
fingerprint = engine.key_fingerprint(api_key)
openai_ready = engine.OPENAI_AVAILABLE and api_key is not None

with st.sidebar:
    st.header("Status")
    st.write(f"OpenAI API key: **{'Available' if api_key else 'Not set'}**")
    st.write(f"openai package: {'installed' if engine.OPENAI_AVAILABLE else 'missing'}")
    st.write(f"chromadb package: {'installed' if engine.CHROMA_AVAILABLE else 'missing'}")
    st.write(f"Tickets loaded: **{len(df)}** across {df['category'].nunique()} categories")
    if data_warnings:
        st.subheader("Data warnings")
        for w in data_warnings:
            st.warning(w)
    if st.button("Reload data & rebuild index"):
        st.cache_data.clear()
        st.cache_resource.clear()
        for k in [k for k in st.session_state if str(k).startswith("semantic_failed::")]:
            del st.session_state[k]
        st.session_state.pop("result", None)
        st.rerun()

chat_client, semantic, setup_warnings = None, None, []
if openai_ready:
    try:
        chat_client = cached_client(fingerprint, api_key)
    except Exception as exc:
        setup_warnings.append(f"Couldn't create the OpenAI client: {engine.safe_error(exc, api_key)}")

if chat_client is not None and engine.CHROMA_AVAILABLE:
    fkey = failure_key(signature, fingerprint)
    if fkey in st.session_state:
        setup_warnings.append(f"Semantic index unavailable this session: {st.session_state[fkey]} Using TF-IDF. Fix the problem, then click 'Reload data & rebuild index'.")
    else:
        try:
            semantic = cached_semantic(signature, engine.EMBED_MODEL, fingerprint, df, chat_client)
        except Exception as exc:
            st.session_state[fkey] = msg = engine.safe_error(exc, api_key)
            setup_warnings.append(f"Semantic indexing failed, using TF-IDF: {msg}")
elif chat_client is not None:
    setup_warnings.append("chromadb isn't available, so retrieval uses TF-IDF.")

if not openai_ready:
    st.info("Base Mode: TF-IDF retrieval and a template reply. Set OPENAI_API_KEY for full RAG mode.")

query = st.text_area("Customer message", value="I was charged twice for the same order.", height=120, max_chars=engine.MAX_QUERY_CHARS)

if st.button("Generate draft", type="primary"):
    try:
        with st.spinner("Retrieving similar tickets and drafting a reply..."):
            result = engine.run_pipeline(query, tfidf=tfidf, semantic=semantic, chat_client=chat_client,
                                         api_key=api_key, extra_warnings=setup_warnings)
        st.session_state["result"] = result
        st.session_state["draft"] = result.draft
    except engine.QueryError as exc:
        st.session_state.pop("result", None)
        st.warning(str(exc))

result = st.session_state.get("result")
if result is not None:
    st.markdown(f"**Pipeline used:** {result.pipeline}")
    for w in result.warnings:
        st.warning(w)
    if result.low_confidence:
        if result.injection_flagged:
            st.warning("Low confidence: the message was flagged as a possible prompt injection. Check before sending.")
        else:
            st.warning(f"Low confidence: the best match scored {result.top_score:.3f}, below the {result.threshold:.2f} "
                       f"threshold for {result.retriever_name}. Check before sending.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Draft reply (editable)")
        draft = st.text_area("Draft", key="draft", height=320, label_visibility="collapsed")
        valid, invalid = engine.check_citations(draft, [r["ticket_id"] for r in result.results])
        if invalid:
            st.error("These citations don't match any retrieved ticket: " + ", ".join(f"[{c}]" for c in invalid) + ". Remove or fix them before sending.")
        if not valid:
            st.caption("The draft currently cites no retrieved ticket.")
    with right:
        st.subheader("Evidence cited in the draft")
        cited = [r for r in result.results if r["ticket_id"] in valid]
        for r in cited:
            st.markdown(f"**{safe_md(r['ticket_id'])}** · {safe_md(r['issue'])}")
            st.caption(safe_md(r["resolution"]))
        if not cited:
            st.caption("Nothing cited yet.")

    st.subheader(f"Retrieved similar tickets (top {len(result.results)})")
    for r in result.results:
        with st.expander(f"{safe_md(r['ticket_id'])} · {safe_md(r['issue'])} · similarity {r['score']:.3f}"):
            st.markdown(f"**Category:** {safe_md(r['category'])}")
            st.markdown(f"**Customer message:** {safe_md(r['customer_message'])}")
            st.markdown(f"**Resolution:** {safe_md(r['resolution'])}")
