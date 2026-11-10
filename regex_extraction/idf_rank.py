from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

from regex_extraction.extract_regex import extract_from_text
from regex_extraction.loose_match import act_key, build_loose_indexes, expression_key

DEFAULT_TYPE_WEIGHTS: dict[str, float] = {
    "act": 1.0,
    "judge": 0.5,
    "law": 2.0,
    "case": 0.0,
}
DEFAULT_SECTION_WEIGHT = 0.5

_SEC_RE = re.compile(
    r"(?i)\b(?:sections?|subsections?|ss?\.|s\.)\s*([A-Za-z0-9.()-]+)"
)

def corpus_size(cases_dir: Path) -> int:
    return sum(1 for _ in cases_dir.glob("*.txt"))

def idf(n_docs: int, df: int) -> float:
    if df <= 0:
        return 0.0
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

def act_section_key(expr: str) -> str:
    base = act_key(expr)
    m = _SEC_RE.search(expr)
    sec = m.group(1).casefold() if m else ""
    return f"{base}||sec:{sec}"

def build_df(
    inverted: dict[str, dict[str, list[str]]],
    loose_indexes: dict[str, dict[str, list[str]]] | None,
    loose: bool,
) -> dict[str, dict[str, int]]:
    df: dict[str, dict[str, int]] = {}
    if loose and loose_indexes is not None:
        for t, buckets in loose_indexes.items():
            df[t] = {key: len(docs) for key, docs in buckets.items()}
    else:
        for t, index in inverted.items():
            df[t] = {expr: len(docs) for expr, docs in index.items()}
    return df

def build_act_section_index(
    inverted: dict[str, dict[str, list[str]]],
) -> dict[str, list[str]]:
    buckets: dict[str, set[str]] = defaultdict(set)
    for expr, docs in inverted.get("act", {}).items():
        buckets[act_section_key(expr)].update(docs)
    return {k: sorted(v) for k, v in buckets.items()}

def query_keys(
    text: str,
    types: list[str],
    loose: bool,
) -> dict[str, set[str]]:
    found = extract_from_text(text)
    keys: dict[str, set[str]] = {t: set() for t in types}
    for t in types:
        for hit in found[t]:
            expr = hit["text"]
            key = expression_key(t, expr) if loose else expr
            if key:
                keys[t].add(key)
    return keys

def score_candidates_wlog(
    text: str,
    types: list[str],
    inverted: dict[str, dict[str, list[str]]],
    loose_indexes: dict[str, dict[str, list[str]]],
    df_map: dict[str, dict[str, int]],
    n_docs: int,
    exclude_id: str | None,
    *,
    type_weights: dict[str, float] | None = None,
    section_weight: float = DEFAULT_SECTION_WEIGHT,
    act_section_index: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    weights = type_weights or DEFAULT_TYPE_WEIGHTS
    keys = query_keys(text, types, loose=True)
    per_type: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for t, key_set in keys.items():
        if weights.get(t, 0.0) == 0.0:
            continue
        index = loose_indexes.get(t, {})
        dfs = df_map.get(t, {})
        for key in key_set:
            docs = index.get(key)
            if not docs:
                continue
            df = dfs.get(key) or len(docs)
            w = idf(n_docs, df)
            for doc_id in docs:
                if exclude_id is not None and doc_id == exclude_id:
                    continue
                per_type[doc_id][t] += w

    if section_weight and "act" in types:
        if act_section_index is None:
            act_section_index = build_act_section_index(inverted)
        found = extract_from_text(text)
        for hit in found.get("act", []):
            sk = act_section_key(hit["text"])
            docs = act_section_index.get(sk)
            if not docs:
                continue
            w = idf(n_docs, len(docs))
            for doc_id in docs:
                if exclude_id is not None and doc_id == exclude_id:
                    continue
                per_type[doc_id]["__sec__"] += w

    out: dict[str, float] = {}
    for doc_id, td in per_type.items():
        score = 0.0
        for t, val in td.items():
            wt = section_weight if t == "__sec__" else weights.get(t, 0.0)
            if wt:
                score += wt * math.log1p(val)
        if score > 0.0:
            out[doc_id] = score
    return out

def rank_docs(doc_scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))

def rank_query(
    text: str,
    query_id: str,
    types: list[str],
    inverted: dict[str, dict[str, list[str]]],
    *,
    loose_indexes: dict[str, dict[str, list[str]]] | None = None,
    n_docs: int,
    keep_self: bool = False,
    k: int | None = None,
    type_weights: dict[str, float] | None = None,
    section_weight: float = DEFAULT_SECTION_WEIGHT,
    act_section_index: dict[str, list[str]] | None = None,
) -> list[tuple[str, float]]:
    if loose_indexes is None:
        loose_indexes = build_loose_indexes(inverted)
    exclude = None if keep_self else query_id
    df_map = build_df(inverted, loose_indexes, True)
    scored = score_candidates_wlog(
        text,
        types,
        inverted,
        loose_indexes,
        df_map,
        n_docs,
        exclude,
        type_weights=type_weights,
        section_weight=section_weight,
        act_section_index=act_section_index,
    )
    ranked = rank_docs(scored)
    if k is not None:
        return ranked[:k]
    return ranked

def type_only_weights(type_name: str) -> dict[str, float]:
    return {t: (1.0 if t == type_name else 0.0) for t in DEFAULT_TYPE_WEIGHTS}

def rank_query_by_type(
    text: str,
    query_id: str,
    type_name: str,
    inverted: dict[str, dict[str, list[str]]],
    *,
    loose_indexes: dict[str, dict[str, list[str]]] | None = None,
    n_docs: int,
    keep_self: bool = False,
    k: int | None = None,
    act_section_index: dict[str, list[str]] | None = None,
    use_section_bonus: bool = True,
) -> list[tuple[str, float]]:
    if type_name not in DEFAULT_TYPE_WEIGHTS:
        raise ValueError(f"tipo invalido: {type_name}")
    sec_w = DEFAULT_SECTION_WEIGHT if (type_name == "act" and use_section_bonus) else 0.0
    return rank_query(
        text,
        query_id,
        types=list(DEFAULT_TYPE_WEIGHTS),
        inverted=inverted,
        loose_indexes=loose_indexes,
        n_docs=n_docs,
        keep_self=keep_self,
        k=k,
        type_weights=type_only_weights(type_name),
        section_weight=sec_w,
        act_section_index=act_section_index,
    )
