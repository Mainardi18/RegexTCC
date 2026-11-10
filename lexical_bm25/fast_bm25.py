from __future__ import annotations

from array import array
from collections import Counter
from typing import Iterable

import numpy as np
from scipy import sparse

DEFAULT_K1 = 3.0
DEFAULT_B = 1.0

class FastBM25:
    def __init__(
        self,
        corpus_tokens: Iterable[list[str]],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        epsilon: float = 0.25,
    ):
        self.vocab: dict[str, int] = {}
        indptr = array("q", [0])
        indices = array("i")
        data = array("f")
        lengths = array("d")
        for toks in corpus_tokens:
            lengths.append(len(toks))
            for term, n in Counter(toks).items():
                j = self.vocab.get(term)
                if j is None:
                    j = len(self.vocab)
                    self.vocab[term] = j
                indices.append(j)
                data.append(n)
            indptr.append(len(indices))

        self.N = len(lengths)
        num_terms = len(self.vocab)
        self.tf = sparse.csr_matrix(
            (
                np.array(data, dtype=np.float32),
                np.array(indices, dtype=np.int32),
                np.array(indptr, dtype=np.int64),
            ),
            shape=(self.N, num_terms),
        )
        self.dl = np.array(lengths, dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if self.N else 0.0

        df = np.bincount(self.tf.indices, minlength=num_terms)
        idf = np.log(self.N - df + 0.5) - np.log(df + 0.5)
        idf[idf < 0] = epsilon * idf.mean()
        self.idf = idf
        self.set_params(k1, b)

    def set_params(self, k1: float, b: float) -> None:
        self.k1, self.b = k1, b
        row_len = np.repeat(self.dl, np.diff(self.tf.indptr))
        tf = self.tf.data.astype(np.float64)
        w = tf * (k1 + 1) / (tf + k1 * (1 - b + b * row_len / self.avgdl))
        weights = sparse.csr_matrix(
            (w.astype(np.float32), self.tf.indices, self.tf.indptr), shape=self.tf.shape
        )
        self._wt = weights.T.tocsr()

    def queries_from_tokens(self, queries: list[list[str]]) -> sparse.csr_matrix:
        rows, cols, vals = [], [], []
        for r, toks in enumerate(queries):
            for term, n in Counter(toks).items():
                j = self.vocab.get(term)
                if j is not None:
                    rows.append(r)
                    cols.append(j)
                    vals.append(n * self.idf[j])
        return sparse.csr_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)),
            shape=(len(queries), len(self.vocab)),
        )

    def scores(self, q: sparse.csr_matrix) -> np.ndarray:
        return (q @ self._wt).toarray()

    def topk(
        self,
        q: sparse.csr_matrix,
        k: int,
        exclude: np.ndarray | None = None,
        batch: int = 128,
    ) -> np.ndarray:
        nq = q.shape[0]
        k = min(k, self.N - (1 if exclude is not None else 0))
        out = np.empty((nq, k), dtype=np.int32)
        for s in range(0, nq, batch):
            scores = self.scores(q[s : s + batch])
            if exclude is not None:
                ex = exclude[s : s + batch]
                rows = np.nonzero(ex >= 0)[0]
                scores[rows, ex[rows]] = -np.inf
            part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
            out[s : s + scores.shape[0]] = np.take_along_axis(part, order, axis=1)
        return out
