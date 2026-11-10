from __future__ import annotations

import re
from collections import defaultdict

_JUDGE_TITLE_RE = re.compile(
    r"^(?P<name>.+),\s*(?:C\.J\.|JJ\.A\.|J\.A\.|D\.J\.|JJ\.|J\.)\s*$"
)

_ACT_LEAD_RE = re.compile(
    r"^(?:pursuant\s+to|under)\s+",
    re.I,
)
_ACT_SECTION_RE = re.compile(
    r"^(?:sections?|subsections?|paragraphs?|ss?\.|paras?\.?|s\.)\s*"
    r"[A-Za-z0-9.()-]+"
    r"(?:\s*(?:and|,|to)\s*"
    r"(?:sections?|subsections?|paragraphs?|ss?\.|paras?\.?|s\.)?\s*"
    r"[A-Za-z0-9.()-]+)*\s+"
    r"(?:of\s+the\s+)?",
    re.I,
)
_ACT_THE_RE = re.compile(r"^the\s+", re.I)
_ACT_CITE_RE = re.compile(
    r",?\s*(?:S\.C\.|R\.S\.C\.|R\.S\.)\s*\d{4}\s*,?\s*c\.?\s*[\w\d.-]+\.?\s*$",
    re.I,
)
_ACT_ANNEX_RE = re.compile(
    r"^(?:ANNEX\s+Relevant\s+Statutory\s+Provisions\s+)",
    re.I,
)
_CASE_V_RE = re.compile(r"\s+vs?\.?\s+", re.I)
_WS_RE = re.compile(r"\s+")

def _clean(s: str) -> str:
    return _WS_RE.sub(" ", s).strip(" .,;:")

def judge_key(expr: str) -> str:
    m = _JUDGE_TITLE_RE.match(expr.strip())
    name = m.group("name") if m else expr
    return _clean(name).casefold()

def act_key(expr: str) -> str:
    s = _clean(expr)
    s = _ACT_ANNEX_RE.sub("", s)
    s = _ACT_LEAD_RE.sub("", s)
    s = _ACT_SECTION_RE.sub("", s)
    s = _ACT_THE_RE.sub("", s)
    s = _ACT_CITE_RE.sub("", s)
    s = _clean(s)
    return s.casefold()

def case_key(expr: str) -> str:
    s = _CASE_V_RE.sub(" v. ", expr)
    s = _clean(s)
    if s.casefold().startswith("re "):
        s = "Re " + s[3:].lstrip()
    return s.casefold()

def law_key(expr: str) -> str:
    return _clean(expr).casefold()

KEYERS = {
    "judge": judge_key,
    "act": act_key,
    "case": case_key,
    "law": law_key,
}

def expression_key(expr_type: str, expr: str) -> str:
    return KEYERS[expr_type](expr)

def build_loose_indexes(
    inverted: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    loose: dict[str, dict[str, list[str]]] = {}
    for t, index in inverted.items():
        keyer = KEYERS[t]
        buckets: dict[str, set[str]] = defaultdict(set)
        for expr, docs in index.items():
            key = keyer(expr)
            if not key:
                continue
            buckets[key].update(docs)
        loose[t] = {
            key: sorted(docs)
            for key, docs in sorted(
                buckets.items(), key=lambda item: (-len(item[1]), item[0])
            )
        }
    return loose

def docs_for_expression(
    expr_type: str,
    expr: str,
    inverted: dict[str, dict[str, list[str]]],
    loose_indexes: dict[str, dict[str, list[str]]] | None,
    *,
    loose: bool,
) -> list[str]:
    if not loose or loose_indexes is None:
        return list(inverted.get(expr_type, {}).get(expr, []))
    key = expression_key(expr_type, expr)
    if not key:
        return list(inverted.get(expr_type, {}).get(expr, []))
    docs = loose_indexes.get(expr_type, {}).get(key)
    if docs is not None:
        return list(docs)
    return list(inverted.get(expr_type, {}).get(expr, []))
