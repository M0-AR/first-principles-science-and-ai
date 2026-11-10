"""E15: RAG — retrieval-augmented generation, from scratch.

Build a small retrieval stack on 20 Newsgroups: BM25 over raw term
frequencies versus TF-IDF cosine, plus the category-match evaluation
that a RAG system's "grounding" depends on. Verify that lexical
retrieval finds same-topic documents without any neural embedder.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from fsl.common import DATA, ExperimentResult, run_experiment

os.environ.setdefault("SCIKIT_LEARN_DATA", str(DATA / "sklearn"))


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=30000)
        self.vec.fit(docs)
        self.counts = CountVectorizer(vocabulary=self.vec.vocabulary_)
        self.X = self.counts.fit_transform(docs).astype(np.float64)
        self.idf = self.vec.idf_
        self.k1, self.b = k1, b
        self.lens = np.asarray(self.X.sum(axis=1)).ravel()
        self.avg = self.lens.mean()

    def scores(self, query: str) -> np.ndarray:
        idx = self.vec.transform([query]).indices
        if len(idx) == 0:
            return np.zeros(self.X.shape[0])
        tf = self.X[:, idx].toarray()
        denom = tf + self.k1 * (1 - self.b + self.b * self.lens[:, None] / self.avg)
        return ((self.k1 + 1) * tf / denom * self.idf[idx]).sum(axis=1)


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        tr = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                                random_state=0)
        te = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"),
                                random_state=0)
        docs = [d[:2000] for d in tr.data[:6000]]
        labels = tr.target[:6000]

        bm25 = BM25(docs)
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=30000)
        Xtf = tfidf.fit_transform(docs)

        rng = np.random.default_rng(0)
        qidx = rng.choice(len(te.data), size=300, replace=False)
        bm25_r5, tf_r5, bm25_top, tf_top = [], [], [], []
        for i in qidx:
            q = te.data[i][:600]
            lab = te.target[i]
            top = np.argsort(bm25.scores(q))[-5:][::-1]
            bm25_r5.append(np.mean(labels[top] == lab))
            bm25_top.append(labels[top[0]] == lab)

            sims = (Xtf @ tfidf.transform([q]).T).toarray().ravel()
            t5 = np.argsort(sims)[-5:][::-1]
            tf_r5.append(np.mean(labels[t5] == lab))
            tf_top.append(labels[t5[0]] == lab)

        bm25_r5 = float(np.mean(bm25_r5)); tf_r5 = float(np.mean(tf_r5))
        bm25_top = float(np.mean(bm25_top)); tf_top = float(np.mean(tf_top))

        res.check("BM25 finds the right category in the top-5 >35% of the time",
                  bm25_r5 > 0.35, f"recall@5={bm25_r5:.3f} (chance={5/20:.2f})")
        res.check("top-1 retrieved document shares the category >40% of the time",
                  bm25_top > 0.40, f"top-1 category accuracy={bm25_top:.3f} (chance={1/20:.3f})")
        res.check("BM25 is competitive with TF-IDF cosine",
                  bm25_r5 >= 0.85 * tf_r5, f"BM25={bm25_r5:.3f}, TF-IDF={tf_r5:.3f}")

        res.metrics.update({"bm25_recall5": bm25_r5, "tfidf_recall5": tf_r5,
                            "bm25_top1": bm25_top, "tfidf_top1": tf_top,
                            "n_docs": len(docs), "n_queries": len(qidx)})
    return run_experiment("e15_rag", body)
