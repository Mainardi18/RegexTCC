from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.coliee_utils import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_LABELS,
    DEFAULT_MAX_QUERY_TOKENS,
    DEFAULT_PRF_DOCS,
    DEFAULT_PRF_TERMS,
    DEFAULT_RRF_K,
    normalize_doc_id,
)
from lexical_bm25.fast_bm25 import FastBM25
from lexical_bm25.query_ops import build_passage_corpus, rank_with_rrf_pipeline

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "artifacts"
KS = (5, 10, 50, 100, 200, 1000)

def recall_table(
    topk: np.ndarray,
    gold: list[np.ndarray],
) -> dict[int, tuple[float, float, int, int]]:
    out: dict[int, tuple[float, float, int, int]] = {}
    for k in KS:
        macros = []
        th = tg = 0
        for i, g in enumerate(gold):
            hits = set(topk[i, :k].tolist()) & set(g.tolist())
            th += len(hits)
            tg += len(g)
            macros.append(len(hits) / len(g) if len(g) else 0.0)
        out[k] = (float(np.mean(macros)), th / tg if tg else 0.0, th, tg)
    return out

def load_passage_index(
    out_dir: Path,
    corpus: list[list[str]],
    k1: float,
    b: float,
) -> tuple[FastBM25, np.ndarray]:
    pass_path = out_dir / "passages.pkl"
    p2d_path = out_dir / "passage_to_doc.npy"
    if pass_path.is_file() and p2d_path.is_file():
        with pass_path.open("rb") as f:
            passages = pickle.load(f)
        return FastBM25(passages, k1=k1, b=b), np.load(p2d_path)
    passages, p2d = build_passage_corpus(corpus)
    return FastBM25(passages, k1=k1, b=b), p2d

def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--k1", type=float, default=DEFAULT_BM25_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_BM25_B)
    args = parser.parse_args()

    corpus_path = args.out_dir / "tokenized_corpus.pkl"
    doc_ids_path = args.out_dir / "doc_ids.json"
    if not corpus_path.is_file() or not doc_ids_path.is_file():
        raise FileNotFoundError(
            f"Artefatos nao encontrados em {args.out_dir}. "
            "Rode: python lexical_bm25/tokenize_corpus.py"
        )

    meta_path = args.out_dir / "tokenize_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    doc_ids = json.loads(doc_ids_path.read_text(encoding="utf-8"))
    with corpus_path.open("rb") as f:
        corpus = pickle.load(f)

    labels_raw = json.loads(args.labels.read_text(encoding="utf-8"))
    labels = {
        normalize_doc_id(q): {normalize_doc_id(r) for r in rels}
        for q, rels in labels_raw.items()
    }
    id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    qids = [q for q in sorted(labels) if q in id_to_idx]
    query_idx = np.array([id_to_idx[q] for q in qids], dtype=np.int32)
    gold = [
        np.array([id_to_idx[r] for r in labels[q] if r in id_to_idx], dtype=np.int32)
        for q in qids
    ]

    print(f"Artefatos: {args.out_dir}")
    print(f"cases_dir: {meta.get('cases_dir', '?')}")
    print(f"docs={len(doc_ids)}  queries={len(qids)}  k1={args.k1}  b={args.b}")

    t0 = time.perf_counter()
    bm25 = FastBM25(corpus, k1=args.k1, b=args.b)
    print("Indexando passagens...")
    bm25_pass, p2d = load_passage_index(args.out_dir, corpus, args.k1, args.b)
    print(f"passages={bm25_pass.N}")

    topk = rank_with_rrf_pipeline(
        bm25,
        corpus,
        query_idx,
        bm25_pass,
        p2d,
        max_tokens=DEFAULT_MAX_QUERY_TOKENS,
        prf_docs=DEFAULT_PRF_DOCS,
        prf_terms=DEFAULT_PRF_TERMS,
        rrf_k=DEFAULT_RRF_K,
        k=max(KS),
    )
    elapsed = time.perf_counter() - t0

    print(f"index+rank: {elapsed:.1f}s  vocab_docs={len(bm25.vocab)}")
    print()
    print(f"{'k':>6} {'macroR':>10} {'microR':>10} {'hits/gold':>14}")
    print("-" * 44)
    for k, (macro, micro, th, tg) in recall_table(topk, gold).items():
        print(f"{k:>6} {macro:10.4f} {micro:10.4f} {th:>6}/{tg}")

if __name__ == "__main__":
    main()
