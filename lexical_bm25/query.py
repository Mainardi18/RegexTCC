from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleaning.case_cleaning import clean_text
from common.coliee_utils import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_MAX_QUERY_TOKENS,
    DEFAULT_PRF_DOCS,
    DEFAULT_PRF_TERMS,
    DEFAULT_RRF_K,
    exclude_doc,
    load_query_text,
    normalize_doc_id,
)
from lexical_bm25.fast_bm25 import FastBM25
from lexical_bm25.query_ops import (
    build_passage_corpus,
    rank_single_query_rrf,
    tokenize_doc,
)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "artifacts"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consulta BM25 RRF (passagens + idf_u+PRF + acts + full)."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=DEFAULT_BM25_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_BM25_B)
    parser.add_argument("--keep-query-doc", action="store_true")
    args = parser.parse_args()

    query_text, query_doc_id = load_query_text(None, args.query_file, None)
    query_text = clean_text(query_text)

    corpus_path = args.out_dir / "tokenized_corpus.pkl"
    doc_ids_path = args.out_dir / "doc_ids.json"
    if not corpus_path.exists() or not doc_ids_path.exists():
        raise FileNotFoundError(
            "Corpus nao encontrado. Rode: python lexical_bm25/tokenize_corpus.py"
        )

    with corpus_path.open("rb") as f:
        tokenized_corpus = pickle.load(f)
    doc_ids: list[str] = json.loads(doc_ids_path.read_text(encoding="utf-8"))

    bm25 = FastBM25(tokenized_corpus, k1=args.k1, b=args.b)
    raw_toks = tokenize_doc(query_text)

    exclude_idx = None
    if query_doc_id and not args.keep_query_doc:
        excl = normalize_doc_id(query_doc_id)
        if excl in doc_ids:
            exclude_idx = doc_ids.index(excl)

    pass_path = args.out_dir / "passages.pkl"
    p2d_path = args.out_dir / "passage_to_doc.npy"
    if pass_path.is_file() and p2d_path.is_file():
        with pass_path.open("rb") as f:
            passages = pickle.load(f)
        p2d = np.load(p2d_path)
        bm25_pass = FastBM25(passages, k1=args.k1, b=args.b)
    else:
        passages, p2d = build_passage_corpus(tokenized_corpus)
        bm25_pass = FastBM25(passages, k1=args.k1, b=args.b)

    t0 = time.perf_counter()
    ordered = rank_single_query_rrf(
        bm25,
        raw_toks,
        bm25_pass,
        p2d,
        exclude_idx=exclude_idx,
        max_tokens=DEFAULT_MAX_QUERY_TOKENS,
        prf_docs=DEFAULT_PRF_DOCS,
        prf_terms=DEFAULT_PRF_TERMS,
        rrf_k=DEFAULT_RRF_K,
        k=args.k + 5,
    )
    query_s = time.perf_counter() - t0

    raw_hits = [(doc_ids[i], s) for i, s in ordered]
    exclude_id = None if args.keep_query_doc else query_doc_id
    hits = exclude_doc(raw_hits, exclude_id, args.k)

    preview = query_text if len(query_text) <= 120 else query_text[:117] + "..."
    print(f"Query: {preview}")
    if query_doc_id:
        print(
            f"Query docId: {query_doc_id} "
            f"(excluido: {exclude_id is not None})"
        )
    print(f"BM25 RRF: k1={args.k1} b={args.b}  tempo={query_s:.4f}s")
    print(f"Top-{args.k}:")
    print("-" * 60)
    for rank, (doc_id, score) in enumerate(hits, start=1):
        print(f"{rank:2d}. docId={doc_id}\tscore={score:.6f}")

if __name__ == "__main__":
    main()
