import hashlib
import os
import re
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DIM = 256


def fake_embedding(text):
    vec = np.zeros(DIM)
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = np.linalg.norm(vec)
    if norm == 0:
        vec[0], norm = 1.0, 1.0
    return (vec / norm).tolist()


class FakeOpenAI:
    def __init__(self, chat_mode="ok", embed_error=False, api_key="sk-test-SECRETKEY123456"):
        self.chat_mode, self.embed_error, self.api_key = chat_mode, embed_error, api_key
        self.embed_calls = self.chat_calls = 0
        self.last_messages = None
        self.embeddings = SimpleNamespace(create=self._embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

    def _embed(self, model, input):
        self.embed_calls += 1
        if self.embed_error:
            raise RuntimeError(f"Embedding service unavailable (key {self.api_key})")
        return SimpleNamespace(data=[SimpleNamespace(embedding=fake_embedding(t)) for t in input])

    def _chat(self, model, messages, temperature, max_tokens):
        self.chat_calls += 1
        self.last_messages = messages
        if self.chat_mode == "error":
            raise RuntimeError(f"Incorrect API key provided: {self.api_key}")
        content = ""
        if self.chat_mode != "empty":
            first_id = re.search(r'<ticket id="([^"]+)"', messages[1]["content"]).group(1)
            content = f"Hi, we've resolved this before [{first_id}]. Best, Support"
            if self.chat_mode == "bad_citation":
                content += " We'll also refund you within 30 days [TKT-999]."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def tickets_path():
    return os.path.join(ROOT, "sample_tickets.csv")


@pytest.fixture
def df(tickets_path):
    import engine
    return engine.load_tickets(tickets_path)[0]


@pytest.fixture
def tfidf(df):
    import engine
    return engine.TfidfRetriever(df)
