"""Retrieval tests on the 900-product development sample.

The resume test uses a fake encoder (no model). The rest build a real BGE
index of the sample in a temp dir once per session (~1 min on CPU) and are
skipped if the model cannot be loaded (e.g. offline with no HF cache).
"""

import hashlib
import json

import numpy as np
import pytest

from retrieval.build_faiss_index import build_index
from retrieval.config import EMBED_DIM, SAMPLE_CORPUS
from retrieval.corpus import load_products
from retrieval.embed_corpus import embed_corpus

SAMPLE = load_products(SAMPLE_CORPUS)
SAMPLE_ASINS = [json.loads(line)["parent_asin"]
                for line in open(SAMPLE_CORPUS, encoding="utf-8") if line.strip()]


def fake_encoder(texts, batch_size=64):
    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        out[i] = np.random.default_rng(seed).standard_normal(EMBED_DIM)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def quiet(*_args, **_kwargs):
    pass


# ---------- no model needed ----------

def test_embedding_resumes_after_interruption(tmp_path):
    first = embed_corpus(SAMPLE_CORPUS, tmp_path, shard_size=250, max_shards=2,
                         encoder=fake_encoder, log=quiet)
    assert first["completed_shards"] == 2 and not first["finished"]

    calls = []

    def counting_encoder(texts, batch_size=64):
        calls.append(len(texts))
        return fake_encoder(texts)

    final = embed_corpus(SAMPLE_CORPUS, tmp_path, shard_size=250,
                         encoder=counting_encoder, log=quiet)
    assert final["finished"] and final["rows_done"] == 900
    assert calls == [250, 150]  # only the remaining rows were encoded

    build_index(tmp_path, log=quiet)
    assert (tmp_path / "row_to_asin.txt").read_text(encoding="utf-8").split() == SAMPLE_ASINS

    # Fake vectors are a deterministic function of text, so resumed shards
    # must match a clean single run exactly.
    clean = tmp_path / "clean"
    embed_corpus(SAMPLE_CORPUS, clean, shard_size=900, encoder=fake_encoder, log=quiet)
    resumed = np.concatenate([np.load(tmp_path / f"emb_{i:05d}.npy") for i in range(4)])
    assert np.allclose(resumed, np.load(clean / "emb_00000.npy"))


def test_resume_rejects_changed_config(tmp_path):
    embed_corpus(SAMPLE_CORPUS, tmp_path, shard_size=300, max_shards=1,
                 encoder=fake_encoder, log=quiet)
    with pytest.raises(ValueError, match="shard_size"):
        embed_corpus(SAMPLE_CORPUS, tmp_path, shard_size=500,
                     encoder=fake_encoder, log=quiet)


# ---------- real BGE index on the sample ----------

@pytest.fixture(scope="session")
def retriever(tmp_path_factory):
    try:
        from retrieval.embed_corpus import load_encoder
        encoder = load_encoder()
    except Exception as exc:  # pragma: no cover - depends on network / cache
        pytest.skip(f"BGE model unavailable: {exc}")
    from retrieval.semantic_search import SemanticRetriever

    index_dir = tmp_path_factory.mktemp("sample_index")
    embed_corpus(SAMPLE_CORPUS, index_dir, encoder=encoder, log=quiet)
    build_index(index_dir, log=quiet)
    return SemanticRetriever(index_dir, encoder=encoder)


def test_index_covers_sample(retriever):
    assert retriever.size == 900
    assert retriever.row_to_asin == SAMPLE_ASINS


def test_output_format(retriever):
    hits = retriever.search("wooden puzzle for a toddler", top_n=100)
    assert len(hits) == 100
    assert [h["rank"] for h in hits] == list(range(1, 101))
    assert all(set(h) == {"parent_asin", "score", "rank"} for h in hits)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in scores)
    assert len({h["parent_asin"] for h in hits}) == 100
    assert all(h["parent_asin"] in SAMPLE for h in hits)


def test_top_n_larger_than_index(retriever):
    assert len(retriever.search("book", top_n=5000)) == 900


def test_empty_query_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("   ")


def test_title_query_finds_its_product(retriever):
    picks = SAMPLE_ASINS[::90]  # 10 products spread over all three domains
    found = 0
    for asin in picks:
        hits = retriever.search(SAMPLE[asin]["title"], top_n=5)
        found += asin in {h["parent_asin"] for h in hits}
    assert found >= 9, f"only {found}/10 titles retrieved their own product"


@pytest.mark.parametrize("query, domain", [
    ("board game for family game night", "Toys & Games"),
    ("handmade sterling silver necklace", "Handmade Products"),
    ("historical fiction novel", "Books"),
])
def test_queries_hit_the_right_domain(retriever, query, domain):
    hits = retriever.search(query, top_n=10)
    share = sum(SAMPLE[h["parent_asin"]]["domain"] == domain for h in hits) / 10
    assert share >= 0.7, f"{query!r}: only {share:.0%} from {domain}"
