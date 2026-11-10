from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

from lexical_bm25.fast_bm25 import FastBM25

LEGAL_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "by", "with",
        "as", "at", "from", "that", "this", "these", "those", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "not", "no", "nor", "but", "if", "then", "than", "so", "such", "may",
        "shall", "will", "would", "could", "should", "must", "can", "also", "any",
        "all", "each", "other", "into", "upon", "under", "over", "about", "between",
        "through", "which", "who", "whom", "whose", "what", "when", "where", "how",
        "there", "here", "their", "its", "his", "her", "they", "them", "he", "she",
        "it", "we", "our", "you", "your", "i", "my", "court", "courts", "judge",
        "justice", "applicant", "applicants", "respondent", "respondents", "appeal",
        "appeals", "appellant", "appellants", "decision", "decisions", "order",
        "orders", "case", "cases", "matter", "matters", "paragraph", "paragraphs",
        "section", "sections", "subsection", "subsections", "see", "supra", "infra",
        "ibid", "pursuant", "regarding", "respect", "respects", "said", "above",
        "below", "following", "whether", "however", "therefore", "thus", "hence",
        "further", "furthermore", "moreover", "accordingly", "canada", "canadian",
        "federal", "board", "minister", "evidence", "reasons", "issue", "issues",
        "fact", "facts", "law", "legal", "party", "parties", "counsel", "submission",
        "submissions", "argument", "arguments", "application", "applications",
        "proceeding", "proceedings", "hearing", "record", "file", "page", "pages",
        "para", "paras", "ss", "c", "s", "r", "v", "vs", "re",
    }
)

_ACT_SPAN_RE = re.compile(
    r"\b(?:"
    r"(?:[a-z][a-z0-9''\-]+\s+){1,6}act|"
    r"(?:[a-z][a-z0-9''\-]+\s+){1,6}regulations?|"
    r"(?:canada\s+)?(?:labour|criminal|civil|income\s+tax)\s+code|"
    r"(?:federal\s+courts?\s+rules|tax\s+court\s+of\s+canada\s+rules|"
    r"rules\s+of\s+civil\s+procedure)|"
    r"(?:canadian\s+)?charter(?:\s+of\s+rights(?:\s+and\s+freedoms)?)?"
    r")\b"
)
_SECTION_RE = re.compile(
    r"\b(?:sections?|subsections?|paragraphs?|ss?\.|paras?\.?)\s*"
    r"([a-z0-9.()-]+(?:\s*(?:and|,|to)\s*[a-z0-9.()-]+)*)",
    re.I,
)

def drop_stops(tokens: list[str], stops: frozenset[str] = LEGAL_STOPWORDS) -> list[str]:
    return [t for t in tokens if t not in stops and len(t) > 1]

def tokenize_doc(text: str) -> list[str]:
    low = text.lower()
    toks = drop_stops(low.split())
    extra: list[str] = []
    for m in _ACT_SPAN_RE.finditer(low):
        span = re.sub(r"[^a-z0-9]+", "_", m.group(0)).strip("_")
        if span and "_" in span:
            extra.append(span)
    for m in _SECTION_RE.finditer(low):
        num = re.sub(r"[^a-z0-9.]+", "_", m.group(1).lower()).strip("_")
        if num:
            extra.append(f"section_{num}")
    return toks + extra

def act_like_tokens(tokens: list[str]) -> list[str]:
    return [
        t
        for t in tokens
        if t.startswith("section_")
        or t.endswith("_act")
        or "_act_" in t
        or t.endswith("_code")
        or t.endswith("_charter")
        or t.endswith("_regulations")
        or t.endswith("_rules")
    ]

def idf_unique_tokens(
    tokens: list[str],
    n: int | None,
    idf: np.ndarray,
    vocab: dict[str, int],
) -> list[str]:
    best: dict[str, float] = {}
    order: dict[str, int] = {}
    for i, t in enumerate(tokens):
        if t not in order:
            order[t] = i
        j = vocab.get(t)
        best[t] = float(idf[j]) if j is not None else -1.0
    ranked = sorted(best, key=lambda t: (-best[t], order[t]))
    take = len(ranked) if n is None or n <= 0 else n
    return sorted(ranked[:take], key=lambda t: order[t])

def prepare_query_tokens(
    tokens: list[str],
    bm25: FastBM25,
    *,
    mode: str = "idf_unique",
    max_tokens: int | None = 300,
) -> list[str]:
    if mode == "full":
        return tokens
    if mode == "idf_u_plus_acts":
        base = idf_unique_tokens(tokens, max_tokens, bm25.idf, bm25.vocab)
        return list(dict.fromkeys(base + act_like_tokens(tokens)))
    if mode == "idf_unique":
        return idf_unique_tokens(tokens, max_tokens, bm25.idf, bm25.vocab)
    raise ValueError(f"query-mode desconhecido: {mode}")

def _inv_vocab(bm25: FastBM25) -> dict[int, str]:
    inv = getattr(bm25, "_inv_vocab", None)
    if inv is None:
        inv = {j: t for t, j in bm25.vocab.items()}
        bm25._inv_vocab = inv
    return inv

def doc_terms(bm25: FastBM25, doc_i: int) -> list[str]:
    inv = _inv_vocab(bm25)
    return [inv[int(j)] for j in bm25.tf.getrow(doc_i).indices]

def prf_expand_queries(
    bm25: FastBM25,
    base_queries: list[list[str]],
    seed_topk: np.ndarray,
    *,
    fb_docs: int = 3,
    fb_terms: int = 50,
    exclude: np.ndarray | None = None,
) -> list[list[str]]:
    expanded: list[list[str]] = []
    for qi, qtoks in enumerate(base_queries):
        qset = set(qtoks)
        scores: Counter[str] = Counter()
        for rank in range(min(fb_docs, seed_topk.shape[1])):
            di = int(seed_topk[qi, rank])
            if exclude is not None and di == int(exclude[qi]):
                continue
            for t in doc_terms(bm25, di):
                if t in qset or t in LEGAL_STOPWORDS or len(t) <= 1:
                    continue
                j = bm25.vocab.get(t)
                if j is None:
                    continue
                scores[t] += float(bm25.idf[j])
        add = [t for t, _ in scores.most_common(fb_terms)]
        expanded.append(qtoks + add)
    return expanded

def _rank_docs(
    bm25: FastBM25,
    corpus: list[list[str]],
    query_idx: np.ndarray,
    *,
    mode: str,
    max_tokens: int | None,
    prf_docs: int,
    prf_terms: int,
    k: int,
    batch: int,
) -> np.ndarray:
    queries = [
        prepare_query_tokens(corpus[int(i)], bm25, mode=mode, max_tokens=max_tokens)
        for i in query_idx
    ]
    qmat = bm25.queries_from_tokens(queries)
    if prf_docs > 0 and prf_terms > 0:
        seed = bm25.topk(qmat, k=max(prf_docs, 20), exclude=query_idx, batch=batch)
        queries = prf_expand_queries(
            bm25, queries, seed, fb_docs=prf_docs, fb_terms=prf_terms, exclude=query_idx
        )
        qmat = bm25.queries_from_tokens(queries)
    return bm25.topk(qmat, k=k, exclude=query_idx, batch=batch)

def build_passage_corpus(
    corpus: list[list[str]],
    window: int = 300,
    stride: int = 200,
) -> tuple[list[list[str]], np.ndarray]:
    passages: list[list[str]] = []
    p2d: list[int] = []
    for di, toks in enumerate(corpus):
        if len(toks) <= window:
            passages.append(toks)
            p2d.append(di)
            continue
        for start in range(0, len(toks), stride):
            chunk = toks[start : start + window]
            if not chunk:
                break
            passages.append(chunk)
            p2d.append(di)
            if start + window >= len(toks):
                break
    return passages, np.array(p2d, dtype=np.int32)

def _passage_topk_to_doc_topk(
    p_topk: np.ndarray,
    p2d: np.ndarray,
    query_idx: np.ndarray,
    k: int,
) -> np.ndarray:
    nq = p_topk.shape[0]
    out = np.empty((nq, k), dtype=np.int32)
    for qi in range(nq):
        seen: set[int] = set()
        docs: list[int] = []
        qex = int(query_idx[qi])
        for p in p_topk[qi]:
            d = int(p2d[int(p)])
            if d == qex or d in seen or d < 0:
                continue
            seen.add(d)
            docs.append(d)
            if len(docs) >= k:
                break
        while len(docs) < k:
            docs.append(-1)
        out[qi] = docs
    return out

def rrf_fuse(
    rankings: list[np.ndarray],
    k: int,
    rrf_k: int = 60,
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    nq = rankings[0].shape[0]
    out = np.empty((nq, k), dtype=np.int32)
    for qi in range(nq):
        scores: dict[int, float] = defaultdict(float)
        for rank in rankings:
            for r, d in enumerate(rank[qi]):
                di = int(d)
                if di < 0:
                    continue
                if exclude is not None and di == int(exclude[qi]):
                    continue
                scores[di] += 1.0 / (rrf_k + r + 1)
        ordered = sorted(scores, key=lambda d: -scores[d])[:k]
        if len(ordered) < k:
            ordered += [-1] * (k - len(ordered))
        out[qi] = ordered
    return out

def rank_with_rrf_pipeline(
    bm25_docs: FastBM25,
    corpus: list[list[str]],
    query_idx: np.ndarray,
    bm25_pass: FastBM25,
    p2d: np.ndarray,
    *,
    max_tokens: int = 300,
    prf_docs: int = 3,
    prf_terms: int = 50,
    rrf_k: int = 60,
    k: int = 1000,
    batch: int = 64,
) -> np.ndarray:
    q_full = [corpus[int(i)] for i in query_idx]
    pool = min(max(k * 8, 2000), 8000)
    p_topk = bm25_pass.topk(
        bm25_pass.queries_from_tokens(q_full),
        k=pool,
        exclude=None,
        batch=max(16, batch // 2),
    )
    rankings = [
        _passage_topk_to_doc_topk(p_topk, p2d, query_idx, k),
        _rank_docs(
            bm25_docs, corpus, query_idx,
            mode="idf_unique", max_tokens=max_tokens,
            prf_docs=prf_docs, prf_terms=prf_terms, k=k, batch=batch,
        ),
        _rank_docs(
            bm25_docs, corpus, query_idx,
            mode="idf_u_plus_acts", max_tokens=max_tokens,
            prf_docs=0, prf_terms=0, k=k, batch=batch,
        ),
        _rank_docs(
            bm25_docs, corpus, query_idx,
            mode="full", max_tokens=None,
            prf_docs=0, prf_terms=0, k=k, batch=batch,
        ),
    ]
    return rrf_fuse(rankings, k, rrf_k=rrf_k, exclude=query_idx)

def rank_single_query_rrf(
    bm25_docs: FastBM25,
    query_tokens_full: list[str],
    bm25_pass: FastBM25,
    p2d: np.ndarray,
    *,
    exclude_idx: int | None = None,
    max_tokens: int = 300,
    prf_docs: int = 3,
    prf_terms: int = 50,
    rrf_k: int = 60,
    k: int = 100,
) -> list[tuple[int, float]]:
    exclude = (
        np.array([exclude_idx], dtype=np.int32) if exclude_idx is not None else None
    )
    rankings: list[np.ndarray] = []

    p_top = bm25_pass.topk(
        bm25_pass.queries_from_tokens([query_tokens_full]),
        k=min(max(k * 8, 2000), 8000),
        exclude=None,
        batch=1,
    )
    seen: set[int] = set()
    docs: list[int] = []
    for p in p_top[0]:
        d = int(p2d[int(p)])
        if exclude_idx is not None and d == exclude_idx:
            continue
        if d in seen:
            continue
        seen.add(d)
        docs.append(d)
        if len(docs) >= k:
            break
    while len(docs) < k:
        docs.append(-1)
    rankings.append(np.array([docs], dtype=np.int32))

    def doc_rank(mode: str, use_prf: bool) -> np.ndarray:
        q = prepare_query_tokens(
            query_tokens_full,
            bm25_docs,
            mode=mode,
            max_tokens=None if mode == "full" else max_tokens,
        )
        if use_prf and prf_docs > 0 and prf_terms > 0:
            seed = bm25_docs.topk(
                bm25_docs.queries_from_tokens([q]),
                k=max(prf_docs, 20),
                exclude=exclude,
                batch=1,
            )
            q = prf_expand_queries(
                bm25_docs, [q], seed,
                fb_docs=prf_docs, fb_terms=prf_terms, exclude=exclude,
            )[0]
        return bm25_docs.topk(
            bm25_docs.queries_from_tokens([q]), k=k, exclude=exclude, batch=1
        )

    rankings.append(doc_rank("idf_unique", True))
    rankings.append(doc_rank("idf_u_plus_acts", False))
    rankings.append(doc_rank("full", False))

    fused = rrf_fuse(rankings, k, rrf_k=rrf_k, exclude=exclude)
    scores: dict[int, float] = defaultdict(float)
    for rank in rankings:
        for r, d in enumerate(rank[0]):
            di = int(d)
            if di < 0:
                continue
            if exclude_idx is not None and di == exclude_idx:
                continue
            scores[di] += 1.0 / (rrf_k + r + 1)
    return [(int(d), float(scores[int(d)])) for d in fused[0] if int(d) >= 0][:k]
