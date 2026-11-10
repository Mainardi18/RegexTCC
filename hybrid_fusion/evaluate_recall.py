from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.coliee_utils import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_CASES_DIR,
    DEFAULT_LABELS,
    DEFAULT_RRF_K,
    normalize_doc_id,
)
from hybrid_fusion.fuse import rrf_fuse_rankings
from hybrid_fusion.rank import (
    DEFAULT_BM25_OUT,
    batch_bm25_rankings,
    load_hybrid_resources,
    rank_regex_ids,
)
from regex_extraction.extract_regex import DEFAULT_INDEX_DIR

KS = (5, 10, 50, 100, 200, 1000)

def recall_from_id_lists(
    rankings: dict[str, list[str]],
    gold: dict[str, set[str]],
    query_ids: list[str],
) -> dict[int, tuple[float, float, int, int]]:
    out: dict[int, tuple[float, float, int, int]] = {}
    for k in KS:
        macros: list[float] = []
        th = tg = 0
        for qid in query_ids:
            g = gold[qid]
            top = set(rankings[qid][:k])
            hits = top & g
            th += len(hits)
            tg += len(g)
            macros.append(len(hits) / len(g) if g else 0.0)
        out[k] = (float(np.mean(macros)), th / tg if tg else 0.0, th, tg)
    return out

def print_table(title: str, table: dict[int, tuple[float, float, int, int]]) -> None:
    print(title)
    print(f"{'k':>6} {'macroR':>10} {'microR':>10} {'hits/gold':>14}")
    print("-" * 44)
    for k in KS:
        macro, micro, th, tg = table[k]
        print(f"{k:>6} {macro:10.4f} {micro:10.4f} {th:>6}/{tg}")
    print()

def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--bm25-out-dir", type=Path, default=DEFAULT_BM25_OUT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--k1", type=float, default=DEFAULT_BM25_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_BM25_B)
    parser.add_argument("--bm25-pool", type=int, default=1000)
    parser.add_argument("--regex-pool", type=int, default=1000)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument(
        "--w-bm25",
        type=float,
        default=4.0,
        help="Peso RRF do ranking BM25 (padrao 4; 1:1 piora o R@100).",
    )
    parser.add_argument("--w-regex", type=float, default=1.0)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Salva JSON com metricas (padrao: hybrid_fusion/results/hybrid_recall.json).",
    )
    args = parser.parse_args()

    res = load_hybrid_resources(
        bm25_out_dir=args.bm25_out_dir,
        index_dir=args.index_dir,
        cases_dir=args.cases_dir,
        k1=args.k1,
        b=args.b,
    )

    labels_raw = json.loads(args.labels.read_text(encoding="utf-8"))
    gold = {
        normalize_doc_id(q): {normalize_doc_id(r) for r in rels}
        for q, rels in labels_raw.items()
    }
    query_ids = [q for q in sorted(gold) if q in res.id_to_idx]
    if args.max_queries is not None:
        query_ids = query_ids[: args.max_queries]

    pool = max(args.bm25_pool, args.regex_pool, max(KS))
    print(
        f"queries={len(query_ids)}  bm25_pool={args.bm25_pool}  "
        f"regex_pool={args.regex_pool}  w=({args.w_bm25},{args.w_regex})"
    )

    t0 = time.perf_counter()
    print("Ranking BM25 (batch RRF)...")
    bm25_rank = batch_bm25_rankings(
        res, query_ids, k=args.bm25_pool, rrf_k=args.rrf_k
    )
    t_bm25 = time.perf_counter() - t0
    print(f"  BM25: {t_bm25:.1f}s")

    print("Ranking regex IDF + fusao...")
    regex_rank: dict[str, list[str]] = {}
    hybrid_rank: dict[str, list[str]] = {}
    t1 = time.perf_counter()
    for i, qid in enumerate(query_ids, 1):
        path = args.cases_dir / f"{qid}.txt"
        text = path.read_text(encoding="utf-8", errors="replace")
        r_ids = rank_regex_ids(res, text, qid, k=args.regex_pool)
        regex_rank[qid] = r_ids
        fused = rrf_fuse_rankings(
            [bm25_rank[qid], r_ids],
            k=pool,
            rrf_k=args.rrf_k,
            weights=[args.w_bm25, args.w_regex],
        )
        hybrid_rank[qid] = [doc_id for doc_id, _ in fused]
        if i % 400 == 0 or i == 1:
            print(f"  {i}/{len(query_ids)}", flush=True)
    t_regex = time.perf_counter() - t1
    print(f"  regex+fuse: {t_regex:.1f}s  total={time.perf_counter() - t0:.1f}s")
    print()

    tables = {
        "bm25": recall_from_id_lists(bm25_rank, gold, query_ids),
        "regex_wlog": recall_from_id_lists(regex_rank, gold, query_ids),
        "hybrid_rrf": recall_from_id_lists(hybrid_rank, gold, query_ids),
    }
    print_table("BM25-only (RRF interno + passagens)", tables["bm25"])
    print_table("Regex-only (score wlog)", tables["regex_wlog"])
    print_table(
        f"Hybrid RRF (w_bm25={args.w_bm25}, w_regex={args.w_regex})",
        tables["hybrid_rrf"],
    )

    b100 = tables["bm25"][100][0]
    h100 = tables["hybrid_rrf"][100][0]
    print(f"Delta macro R@100 (hybrid - bm25): {h100 - b100:+.4f}")

    save_path = args.save
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "results" / "hybrid_recall.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_queries": len(query_ids),
        "bm25_pool": args.bm25_pool,
        "regex_pool": args.regex_pool,
        "rrf_k": args.rrf_k,
        "w_bm25": args.w_bm25,
        "w_regex": args.w_regex,
        "metrics": {
            name: {
                str(k): {
                    "macro": round(macro, 6),
                    "micro": round(micro, 6),
                    "hits": th,
                    "gold": tg,
                }
                for k, (macro, micro, th, tg) in table.items()
            }
            for name, table in tables.items()
        },
    }
    save_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Salvo: {save_path}")

if __name__ == "__main__":
    main()
