from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.coliee_utils import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_CASES_DIR,
    DEFAULT_RRF_K,
    exclude_doc,
    load_query_text,
)
from hybrid_fusion.rank import (
    DEFAULT_BM25_OUT,
    load_hybrid_resources,
    rank_hybrid_query,
)
from regex_extraction.extract_regex import DEFAULT_INDEX_DIR

def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bm25-pool", type=int, default=1000)
    parser.add_argument("--regex-pool", type=int, default=1000)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--w-bm25", type=float, default=4.0)
    parser.add_argument("--w-regex", type=float, default=1.0)
    parser.add_argument("--bm25-out-dir", type=Path, default=DEFAULT_BM25_OUT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--k1", type=float, default=DEFAULT_BM25_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_BM25_B)
    parser.add_argument("--keep-query-doc", action="store_true")
    parser.add_argument("--show-sources", action="store_true")
    args = parser.parse_args()

    text, query_id = load_query_text(None, args.query_file, None)
    res = load_hybrid_resources(
        bm25_out_dir=args.bm25_out_dir,
        index_dir=args.index_dir,
        cases_dir=args.cases_dir,
        k1=args.k1,
        b=args.b,
    )

    t0 = time.perf_counter()
    result = rank_hybrid_query(
        res,
        query_text_raw=text,
        query_id=query_id,
        k=args.k + 5,
        bm25_pool=args.bm25_pool,
        regex_pool=args.regex_pool,
        rrf_k=args.rrf_k,
        w_bm25=args.w_bm25,
        w_regex=args.w_regex,
        keep_query_doc=args.keep_query_doc,
    )
    elapsed = time.perf_counter() - t0

    exclude_id = None if args.keep_query_doc else query_id
    hits = exclude_doc(result["fused"], exclude_id, args.k)

    bm25_pos = {d: i + 1 for i, d in enumerate(result["bm25"])}
    regex_pos = {d: i + 1 for i, d in enumerate(result["regex"])}

    preview = text if len(text) <= 120 else text[:117] + "..."
    print(f"Query: {preview}")
    if query_id:
        print(f"Query docId: {query_id}")
    print(f"Hybrid RRF fuse w=({args.w_bm25},{args.w_regex})  tempo={elapsed:.3f}s")
    print(f"Top-{args.k}:")
    print("-" * 78)
    for rank, (doc_id, score) in enumerate(hits, start=1):
        line = f"{rank:2d}. docId={doc_id}\tscore={score:.6f}"
        if args.show_sources:
            b = bm25_pos.get(doc_id)
            r = regex_pos.get(doc_id)
            b_s = f"@{b}" if b is not None else "-"
            r_s = f"@{r}" if r is not None else "-"
            line += f"\tbm25={b_s}\tregex={r_s}"
        print(line)

if __name__ == "__main__":
    main()
