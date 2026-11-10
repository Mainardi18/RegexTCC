from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cleaning.case_cleaning import clean_text
from common.coliee_utils import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_CASES_DIR,
    DEFAULT_MAX_QUERY_TOKENS,
    DEFAULT_PRF_DOCS,
    DEFAULT_PRF_TERMS,
    DEFAULT_RRF_K,
    normalize_doc_id,
)
from hybrid_fusion.fuse import rrf_fuse_rankings
from lexical_bm25.fast_bm25 import FastBM25
from lexical_bm25.query_ops import (
    build_passage_corpus,
    rank_single_query_rrf,
    rank_with_rrf_pipeline,
    tokenize_doc,
)
from regex_extraction.extract_regex import DEFAULT_INDEX_DIR, PATTERNS, load_indexes
from regex_extraction.idf_rank import corpus_size, rank_query
from regex_extraction.loose_match import build_loose_indexes

DEFAULT_BM25_OUT = Path(__file__).resolve().parents[1] / "lexical_bm25" / "artifacts"

@dataclass
class HybridResources:
    doc_ids: list[str]
    id_to_idx: dict[str, int]
    corpus: list[list[str]]
    bm25: FastBM25
    bm25_pass: FastBM25
    p2d: np.ndarray
    inverted: dict[str, dict[str, list[str]]]
    loose_indexes: dict[str, dict[str, list[str]]]
    n_docs: int
    types: list[str]
    cases_dir: Path
    k1: float
    b: float

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

def load_hybrid_resources(
    *,
    bm25_out_dir: Path = DEFAULT_BM25_OUT,
    index_dir: Path = DEFAULT_INDEX_DIR,
    cases_dir: Path = DEFAULT_CASES_DIR,
    k1: float = DEFAULT_BM25_K1,
    b: float = DEFAULT_BM25_B,
    types: list[str] | None = None,
) -> HybridResources:
    corpus_path = bm25_out_dir / "tokenized_corpus.pkl"
    doc_ids_path = bm25_out_dir / "doc_ids.json"
    if not corpus_path.is_file() or not doc_ids_path.is_file():
        raise FileNotFoundError(
            f"Artefatos BM25 nao encontrados em {bm25_out_dir}. "
            "Rode: python lexical_bm25/tokenize_corpus.py"
        )

    doc_ids: list[str] = json.loads(doc_ids_path.read_text(encoding="utf-8"))
    with corpus_path.open("rb") as f:
        corpus = pickle.load(f)

    types = list(types) if types is not None else list(PATTERNS)
    inverted = load_indexes(index_dir, types)
    loose_indexes = build_loose_indexes(inverted)

    bm25 = FastBM25(corpus, k1=k1, b=b)
    bm25_pass, p2d = load_passage_index(bm25_out_dir, corpus, k1, b)

    return HybridResources(
        doc_ids=doc_ids,
        id_to_idx={d: i for i, d in enumerate(doc_ids)},
        corpus=corpus,
        bm25=bm25,
        bm25_pass=bm25_pass,
        p2d=p2d,
        inverted=inverted,
        loose_indexes=loose_indexes,
        n_docs=corpus_size(cases_dir),
        types=types,
        cases_dir=cases_dir,
        k1=k1,
        b=b,
    )

def rank_bm25_ids(
    res: HybridResources,
    query_tokens: list[str],
    *,
    exclude_idx: int | None,
    k: int,
    max_tokens: int = DEFAULT_MAX_QUERY_TOKENS,
    prf_docs: int = DEFAULT_PRF_DOCS,
    prf_terms: int = DEFAULT_PRF_TERMS,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[str]:
    ordered = rank_single_query_rrf(
        res.bm25,
        query_tokens,
        res.bm25_pass,
        res.p2d,
        exclude_idx=exclude_idx,
        max_tokens=max_tokens,
        prf_docs=prf_docs,
        prf_terms=prf_terms,
        rrf_k=rrf_k,
        k=k,
    )
    return [res.doc_ids[i] for i, _ in ordered]

def rank_regex_ids(
    res: HybridResources,
    query_text: str,
    query_id: str,
    *,
    k: int,
) -> list[str]:
    ranked = rank_query(
        query_text,
        query_id,
        res.types,
        res.inverted,
        loose_indexes=res.loose_indexes,
        n_docs=res.n_docs,
        keep_self=False,
        k=k,
    )
    return [doc_id for doc_id, _ in ranked]

def fuse_hybrid(
    bm25_ranking: list[str],
    regex_ranking: list[str],
    *,
    k: int,
    rrf_k: int = DEFAULT_RRF_K,
    w_bm25: float = 1.0,
    w_regex: float = 1.0,
) -> list[tuple[str, float]]:
    return rrf_fuse_rankings(
        [bm25_ranking, regex_ranking],
        k=k,
        rrf_k=rrf_k,
        weights=[w_bm25, w_regex],
    )

def rank_hybrid_query(
    res: HybridResources,
    *,
    query_text_raw: str,
    query_id: str | None,
    k: int = 100,
    bm25_pool: int = 1000,
    regex_pool: int = 1000,
    rrf_k: int = DEFAULT_RRF_K,
    w_bm25: float = 4.0,
    w_regex: float = 1.0,
    keep_query_doc: bool = False,
) -> dict:
    qid = normalize_doc_id(query_id) if query_id else None
    clean = clean_text(query_text_raw)
    tokens = tokenize_doc(clean)

    exclude_idx = None
    if qid and not keep_query_doc and qid in res.id_to_idx:
        exclude_idx = res.id_to_idx[qid]

    raw_for_regex = query_text_raw
    if qid:
        path = res.cases_dir / f"{qid}.txt"
        if path.is_file():
            raw_for_regex = path.read_text(encoding="utf-8", errors="replace")

    bm25_ranking = rank_bm25_ids(
        res, tokens, exclude_idx=exclude_idx, k=bm25_pool, rrf_k=rrf_k
    )
    regex_ranking = rank_regex_ids(res, raw_for_regex, qid or "", k=regex_pool)
    fused = fuse_hybrid(
        bm25_ranking,
        regex_ranking,
        k=k,
        rrf_k=rrf_k,
        w_bm25=w_bm25,
        w_regex=w_regex,
    )
    return {
        "fused": fused,
        "bm25": bm25_ranking,
        "regex": regex_ranking,
    }

def batch_bm25_rankings(
    res: HybridResources,
    query_ids: list[str],
    *,
    k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[str, list[str]]:
    query_idx = np.array([res.id_to_idx[q] for q in query_ids], dtype=np.int32)
    topk = rank_with_rrf_pipeline(
        res.bm25,
        res.corpus,
        query_idx,
        res.bm25_pass,
        res.p2d,
        max_tokens=DEFAULT_MAX_QUERY_TOKENS,
        prf_docs=DEFAULT_PRF_DOCS,
        prf_terms=DEFAULT_PRF_TERMS,
        rrf_k=rrf_k,
        k=k,
    )
    out: dict[str, list[str]] = {}
    for i, qid in enumerate(query_ids):
        ids: list[str] = []
        for di in topk[i]:
            di_i = int(di)
            if di_i < 0:
                continue
            ids.append(res.doc_ids[di_i])
        out[qid] = ids
    return out
