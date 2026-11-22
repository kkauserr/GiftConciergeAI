import argparse
import json
import os
import pickle
import re
import time

import bm25s
import numpy as np
import pandas as pd
import Stemmer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

META_COLUMNS = [
    "parent_asin", "domain", "title", "price_num", "average_rating", "rating_number",
    "brand_or_author", "main_category", "specific_category", "categories",
    "age_min", "age_max", "review_count", "average_review_rating",
]

_stemmer = Stemmer.Stemmer("english")


def iter_corpus(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_texts_and_meta(path, limit=None):
    texts, meta = [], []
    for row in iter_corpus(path, limit):
        if not row.get("parent_asin") or not row.get("bm25_text"):
            continue
        texts.append(row["bm25_text"])
        m = {c: row.get(c) for c in META_COLUMNS}
        m["categories"] = " > ".join(m["categories"] or [])
        meta.append(m)
    return texts, pd.DataFrame(meta)


def corpus_stats(meta):
    rated = meta[meta["rating_number"].fillna(0) > 0]
    counts = rated["rating_number"].astype(float)
    return {
        "n_products": int(len(meta)),
        "mean_rating": float(rated["average_rating"].mean()),
        "prior_count": float(max(counts.median(), 1.0)),
        "pop_cap": float(np.log1p(counts.quantile(0.99))),
    }


_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
_STOPWORDS = set(bm25s.stopwords.STOPWORDS_EN)


def stem_analyzer(text):
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
    return _stemmer.stemWords(tokens)


def keyword_query(constraints, interest_boost=2):
    parts = list(constraints.get("interests") or []) * interest_boost
    parts += list(constraints.get("keywords") or [])
    if constraints.get("occasion"):
        parts.append(constraints["occasion"])
    return " ".join(parts)


class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tokenizer = bm25s.tokenization.Tokenizer(stemmer=_stemmer, stopwords="en")
        self.model = None
        self.asins = None

    def build(self, texts, asins):
        tokens = self.tokenizer.tokenize(texts, return_as="tuple", show_progress=False)
        self.model = bm25s.BM25(k1=self.k1, b=self.b)
        self.model.index(tokens, show_progress=False)
        self.asins = np.array(asins)
        return self

    def search(self, query, k=200):
        if not query.strip():
            return []
        q = self.tokenizer.tokenize([query], return_as="ids", update_vocab=False, show_progress=False)
        if all(t == 0 for t in q[0]):
            return []
        k = min(k, len(self.asins))
        idx, scores = self.model.retrieve(q, k=k, show_progress=False)
        return [(str(self.asins[i]), float(s)) for i, s in zip(idx[0], scores[0]) if s > 0]

    def save(self, out_dir):
        d = os.path.join(out_dir, "bm25")
        os.makedirs(d, exist_ok=True)
        self.model.save(d)
        self.tokenizer.save_vocab(d)
        self.tokenizer.save_stopwords(d)
        np.save(os.path.join(d, "asins.npy"), self.asins)

    @classmethod
    def load(cls, out_dir):
        d = os.path.join(out_dir, "bm25")
        r = cls()
        r.model = bm25s.BM25.load(d)
        r.tokenizer.load_vocab(d)
        r.tokenizer.load_stopwords(d)
        r.asins = np.load(os.path.join(d, "asins.npy"), allow_pickle=True)
        r.k1, r.b = r.model.k1, r.model.b
        return r


class TfidfRetriever:
    def __init__(self, max_features=500_000):
        self.vectorizer = TfidfVectorizer(
            analyzer=stem_analyzer, sublinear_tf=True, min_df=2, max_features=max_features
        )
        self.matrix = None
        self.asins = None

    def build(self, texts, asins):
        self.matrix = self.vectorizer.fit_transform(texts).tocsr()
        self.asins = np.array(asins)
        return self

    def search(self, query, k=200):
        q = self.vectorizer.transform([query])
        if q.nnz == 0:
            return []
        scores = (self.matrix @ q.T).toarray().ravel()
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(str(self.asins[i]), float(scores[i])) for i in top if scores[i] > 0]

    def save(self, out_dir):
        d = os.path.join(out_dir, "tfidf")
        os.makedirs(d, exist_ok=True)
        sparse.save_npz(os.path.join(d, "matrix.npz"), self.matrix)
        with open(os.path.join(d, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        np.save(os.path.join(d, "asins.npy"), self.asins)

    @classmethod
    def load(cls, out_dir):
        d = os.path.join(out_dir, "tfidf")
        r = cls()
        r.matrix = sparse.load_npz(os.path.join(d, "matrix.npz")).tocsr()
        with open(os.path.join(d, "vectorizer.pkl"), "rb") as f:
            r.vectorizer = pickle.load(f)
        r.asins = np.load(os.path.join(d, "asins.npy"), allow_pickle=True)
        return r


class Catalog:
    def __init__(self, meta, stats):
        self.meta = meta.set_index("parent_asin", drop=False)
        self.stats = stats

    def get(self, asins):
        return self.meta.loc[[a for a in asins if a in self.meta.index]]

    @classmethod
    def load(cls, out_dir):
        meta = pd.read_parquet(os.path.join(out_dir, "catalog.parquet"))
        with open(os.path.join(out_dir, "stats.json")) as f:
            stats = json.load(f)
        return cls(meta, stats)


def build_all(corpus_path, out_dir, limit=None, skip_tfidf=False):
    t0 = time.time()
    texts, meta = load_texts_and_meta(corpus_path, limit)
    print(f"Loaded {len(texts):,} products in {time.time() - t0:.1f}s")
    os.makedirs(out_dir, exist_ok=True)

    meta.to_parquet(os.path.join(out_dir, "catalog.parquet"), index=False)
    stats = corpus_stats(meta)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("Corpus stats:", stats)

    asins = meta["parent_asin"].tolist()
    t0 = time.time()
    BM25Retriever().build(texts, asins).save(out_dir)
    print(f"BM25 index built in {time.time() - t0:.1f}s")

    if not skip_tfidf:
        t0 = time.time()
        TfidfRetriever().build(texts, asins).save(out_dir)
        print(f"TF-IDF index built in {time.time() - t0:.1f}s")
    print(f"Saved to {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Build or query the BM25 / TF-IDF baseline")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--corpus", default="DATA/samples/sample_corpus_v2.jsonl")
    b.add_argument("--out", default="indexes/sample")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--skip-tfidf", action="store_true")

    q = sub.add_parser("search")
    q.add_argument("query")
    q.add_argument("--index", default="indexes/sample")
    q.add_argument("--method", choices=["bm25", "tfidf"], default="bm25")
    q.add_argument("-k", type=int, default=10)

    args = p.parse_args()
    from CODE.retrieval import bm25_baseline as mod  # keeps pickled analyzer importable
    if args.cmd == "build":
        mod.build_all(args.corpus, args.out, args.limit, args.skip_tfidf)
    else:
        r = (mod.BM25Retriever if args.method == "bm25" else mod.TfidfRetriever).load(args.index)
        cat = mod.Catalog.load(args.index)
        for rank, (asin, score) in enumerate(r.search(args.query, args.k), 1):
            row = cat.meta.loc[asin]
            print(f"{rank:>2}. {score:7.3f}  ${row.price_num:<8.2f} [{row.domain}] {row.title[:80]}")


if __name__ == "__main__":
    main()
