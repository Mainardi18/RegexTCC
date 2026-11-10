from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.coliee_utils import DEFAULT_CASES_DIR, DEFAULT_LABELS, normalize_doc_id
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

KS = (5, 10, 50, 100, 200, 1000)
TYPES = ("judge", "law", "act", "case")

def recall_at_k(ranking: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    return len(set(ranking[:k]) & gold) / len(gold)

def dcg_at_k(rels: list[float], k: int) -> float:
    s = 0.0
    for i, r in enumerate(rels[:k], start=1):
        if r > 0:
            s += r / math.log2(i + 1)
    return s

def ndcg_at_k(ranking: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return float("nan")
    rels = [1.0 if d in gold else 0.0 for d in ranking[:k]]
    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k([1.0] * min(len(gold), k), k)
    return dcg / idcg if idcg > 0 else 0.0

def summarize(
    rankings: dict[str, list[str]],
    gold: dict[str, set[str]],
    query_ids: list[str],
) -> dict[str, dict[int, dict[str, float | int]]]:
    out: dict[str, dict[int, dict[str, float | int]]] = {}
    for k in KS:
        macros_r: list[float] = []
        ndcgs: list[float] = []
        th = tg = 0
        empty = 0
        for qid in query_ids:
            g = gold[qid]
            rank = rankings[qid]
            if not rank:
                empty += 1
            if not g:
                continue
            r = recall_at_k(rank, g, k)
            macros_r.append(r)
            ndcgs.append(ndcg_at_k(rank, g, k))
            th += len(set(rank[:k]) & g)
            tg += len(g)
        out[str(k)] = {
            "macro_recall": round(float(np.nanmean(macros_r)) if macros_r else 0.0, 6),
            "micro_recall": round(th / tg if tg else 0.0, 6),
            "ndcg": round(float(np.nanmean(ndcgs)) if ndcgs else 0.0, 6),
            "hits": th,
            "gold": tg,
            "queries_empty_ranking": empty,
        }
    return out

def print_block(title: str, metrics: dict[str, dict[int, dict[str, float | int]]]) -> None:
    print(title)
    print(f"{'k':>6} {'macroR':>10} {'microR':>10} {'NDCG':>10} {'hits/gold':>14}")
    print("-" * 56)
    for k in KS:
        m = metrics[str(k)]
        print(
            f"{k:>6} {m['macro_recall']:10.4f} {m['micro_recall']:10.4f} "
            f"{m['ndcg']:10.4f} {m['hits']:>6}/{m['gold']}"
        )
    print()

def main() -> None:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--k-pool", type=int, default=1000)
    parser.add_argument(
        "--types",
        default=",".join(TYPES),
        help="Tipos a avaliar (padrao: judge,law,act,case).",
    )
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    invalid = [t for t in types if t not in PATTERNS]
    if invalid:
        raise SystemExit(f"Tipos invalidos: {invalid}. Use: {list(PATTERNS)}")

    inverted = load_indexes(args.index_dir, list(PATTERNS))
    loose = build_loose_indexes(inverted)
    sec_index = build_act_section_index(inverted)
    n_docs = corpus_size(args.cases_dir)

    labels_raw = json.loads(args.labels.read_text(encoding="utf-8"))
    gold = {
        normalize_doc_id(q): {normalize_doc_id(r) for r in rels}
        for q, rels in labels_raw.items()
    }
    query_ids = sorted(gold)
    if args.max_queries is not None:
        query_ids = query_ids[: args.max_queries]

    print(f"queries={len(query_ids)}  k_pool={args.k_pool}  types={types}")
    t0 = time.perf_counter()

    by_type: dict[str, dict[str, list[str]]] = {t: {} for t in types}
    combined: dict[str, list[str]] = {}

    for i, qid in enumerate(query_ids, 1):
        text = (args.cases_dir / f"{qid}.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        for t in types:
            ranked = rank_query_by_type(
                text,
                qid,
                t,
                inverted,
                loose_indexes=loose,
                n_docs=n_docs,
                keep_self=False,
                k=args.k_pool,
                act_section_index=sec_index,
                use_section_bonus=True,
            )
            by_type[t][qid] = [d for d, _ in ranked]

        ranked_all = rank_query(
            text,
            qid,
            list(PATTERNS),
            inverted,
            loose_indexes=loose,
            n_docs=n_docs,
            keep_self=False,
            k=args.k_pool,
            type_weights=DEFAULT_TYPE_WEIGHTS,
            section_weight=DEFAULT_SECTION_WEIGHT,
            act_section_index=sec_index,
        )
        combined[qid] = [d for d, _ in ranked_all]

        if i % 200 == 0 or i == 1:
            print(f"  {i}/{len(query_ids)}", flush=True)

    print(f"tempo: {time.perf_counter() - t0:.1f}s\n")

    results: dict[str, dict] = {}
    for t in types:
        metrics = summarize(by_type[t], gold, query_ids)
        empty100 = sum(1 for q in query_ids if not by_type[t][q])
        print_block(f"Regex rank — so {t}  (queries sem candidatos: {empty100})", metrics)
        results[t] = metrics

    metrics_all = summarize(combined, gold, query_ids)
    print_block("Regex rank — combinado wlog (4 tipos)", metrics_all)
    results["combined_wlog"] = metrics_all

    print("Resumo @100:")
    print(f"{'modelo':<18} {'macroR':>10} {'NDCG':>10} {'hits':>10}")
    print("-" * 52)
    for name in list(types) + ["combined_wlog"]:
        m = results[name]["100"]
        print(
            f"{name:<18} {m['macro_recall']:10.4f} {m['ndcg']:10.4f} "
            f"{m['hits']:>6}/{m['gold']}"
        )

    save_path = args.save
    if save_path is None:
        save_path = (
            Path(__file__).resolve().parent / "results" / "regex_rank_by_type.json"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_queries": len(query_ids),
        "k_pool": args.k_pool,
        "types": types,
        "score": "wlog_type_only",
        "combined_weights": DEFAULT_TYPE_WEIGHTS,
        "section_weight_for_act": DEFAULT_SECTION_WEIGHT,
        "metrics": results,
    }
    save_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSalvo: {save_path}")

if __name__ == "__main__":
    main()
