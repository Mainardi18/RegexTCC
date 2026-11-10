from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.coliee_utils import DEFAULT_CASES_DIR, exclude_doc, load_query_text
from regex_extraction.extract_regex import DEFAULT_INDEX_DIR, PATTERNS, load_indexes
from regex_extraction.idf_rank import (
    DEFAULT_SECTION_WEIGHT,
    DEFAULT_TYPE_WEIGHTS,
    build_act_section_index,
    corpus_size,
    rank_query,
    rank_query_by_type,
)
from regex_extraction.loose_match import build_loose_indexes

def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--type",
        choices=list(PATTERNS),
        default=None,
        help="Se definido, ranking só com esse tipo.",
    )
    parser.add_argument("--keep-self", action="store_true")
    args = parser.parse_args()

    text, query_id = load_query_text(None, args.query_file, None)
    qid = query_id or args.query_file.stem
    types = list(PATTERNS)
    inverted = load_indexes(args.index_dir, types)
    loose = build_loose_indexes(inverted)
    sec_index = build_act_section_index(inverted)
    n_docs = corpus_size(args.cases_dir)

    t0 = time.perf_counter()
    if args.type:
        ranked = rank_query_by_type(
            text,
            qid,
            args.type,
            inverted,
            loose_indexes=loose,
            n_docs=n_docs,
            keep_self=args.keep_self,
            k=args.k + 5,
            act_section_index=sec_index,
        )
        label = f"regex tipo={args.type}"
    else:
        ranked = rank_query(
            text,
            qid,
            types,
            inverted,
            loose_indexes=loose,
            n_docs=n_docs,
            keep_self=args.keep_self,
            k=args.k + 5,
            type_weights=DEFAULT_TYPE_WEIGHTS,
            section_weight=DEFAULT_SECTION_WEIGHT,
            act_section_index=sec_index,
        )
        label = "regex combinado wlog"
    elapsed = time.perf_counter() - t0

    exclude_id = None if args.keep_self else qid
    hits = exclude_doc(ranked, exclude_id, args.k)

    preview = text if len(text) <= 120 else text[:117] + "..."
    print(f"Query: {preview}")
    print(f"Query docId: {qid}")
    print(f"{label}  tempo={elapsed:.3f}s")
    print(f"Top-{args.k}:")
    print("-" * 60)
    for i, (doc_id, score) in enumerate(hits, start=1):
        print(f"{i:2d}. docId={doc_id}\tscore={score:.6f}")

if __name__ == "__main__":
    main()
