from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.coliee_utils import DEFAULT_CASES_DIR

REGEX_DIR = Path(__file__).resolve().parent
DEFAULT_MATCHES_OUTPUT = REGEX_DIR / "results" / "regex_matches.json"
DEFAULT_INDEX_DIR = REGEX_DIR / "indexes"

JUDGE_SUFFIX_RE = re.compile(
    r"(?P<names>(?:de\s+|van\s+|von\s+|da\s+|del\s+|le\s+)?"
    r"[A-ZÀ-Ÿ][A-Za-zÀ-ÿ''\-]+"
    r"(?:\s+(?:and|&|de|van|von|da|del|le|[A-ZÀ-Ÿ][A-Za-zÀ-ÿ''\-]+))*)"
    r",\s+(?P<title>C\.J\.|JJ\.A\.|J\.A\.|D\.J\.|JJ\.|J\.)"
)

_JUDGE_PARTICLE = r"(?:de|van|von|da|del|le|du|la)"
_JUDGE_NAME_TOKEN = r"(?:[A-ZÀ-Ÿ]\.|[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'’\-]+)"
JUDGE_TITLE_RE = re.compile(
    r"(?<!\bof\s)(?<!\bof\sthe\s)(?<!\bthe\s)(?<!\bThe\s)"
    r"\b(?P<prefix>(?:Mr\.|Mrs\.|Ms\.|Madam|Madame|Chief|Associate Chief|Deputy|Associate|Senior)\s+)?"
    r"(?:Judge|Justice)\s+"
    r"(?!(?:Canada|Act|Department|System|Minister|Council|Institute|Committee|Advocate"
    r"|Court|Building|Centre|Center|Education|Society|Reform|Branch|Officials?|Services?"
    r"|Canada's|Québec|Quebec|Ontario|Alberta|Manitoba|Saskatchewan"
    r"|Inquiry|Parties|Ordered|Orders?|Clerk|Erred|Errs?|Provided?|Reasons|Failed|Did"
    r"|Was|Is|Has|Had|Should|Must|Judgment|Decision|Review|Issues?|Analysis|Conclusion"
    r"|Delivered|Directed|Held|Found|Made|Considered|Concluded|Determined|Stated|Noted)\b)"
    rf"(?P<name>(?:{_JUDGE_PARTICLE}\s+)?{_JUDGE_NAME_TOKEN}"
    rf"(?:\s+(?:{_JUDGE_PARTICLE}\s+)?{_JUDGE_NAME_TOKEN}){{0,3}})"
)

JUDGE_RE = re.compile(
    rf"(?:{JUDGE_SUFFIX_RE.pattern})|(?:{JUDGE_TITLE_RE.pattern})"
)

LAW_RE = re.compile(
    r"(?P<subject>[A-ZÀ-Ÿ](?:[A-Za-zÀ-ÿ0-9 ,&'/().-]|\xa0|\xad){1,80}?)"
    r"(?:[\s\xa0]*[\-\u2013\u2014][\s\xa0]*|[\s\xa0]+)"
    r"[Tt]opic[\s\xa0]+(?P<num>\d+(?:\.\d+)*)(?!\s*:)"
)

_ACT_STOP = r"(?:This|The|An|That|Said|Each|Such|Any|Every|Former|Previous)"
_INSTRUMENT_NAME = (
    r"[A-ZÀ-Ÿ][A-Za-zÀ-ÿ''\-]+"
    r"(?:\s+(?:and|of|the|for|in|on|[A-ZÀ-Ÿ][A-Za-zÀ-ÿ''\-]+))*"
)

ACT_NAME_RE = re.compile(
    rf"(?!{_ACT_STOP}\s+Act\b){_INSTRUMENT_NAME}\s+Act\b"
)
REGULATION_NAME_RE = re.compile(
    rf"(?!{_ACT_STOP}\s+Regulations?\b){_INSTRUMENT_NAME}\s+Regulations?\b"
)
CODE_RE = re.compile(
    r"\b(?:Canada\s+Labour\s+Code|Labour\s+Relations\s+Code|Labour\s+Code|"
    r"Income\s+Tax\s+Code|Criminal\s+Code|Civil\s+Code)\b"
)
RULES_RE = re.compile(
    r"\b(?:Federal\s+Courts?\s+Rules|Tax\s+Court\s+of\s+Canada\s+Rules|"
    r"Rules\s+of\s+Civil\s+Procedure)\b"
)
CHARTER_RE = re.compile(
    r"\b(?:Canadian\s+)?Charter(?:\s+of\s+Rights(?:\s+and\s+Freedoms)?)?\b"
)

ACT_PREFIX_RE = re.compile(
    r"(?:"
    r"(?:(?:pursuant\s+to|under)\s+)?"
    r"(?:sections?|subsections?|paragraphs?|ss?\.|paras?\.?)\s*"
    r"[A-Za-z0-9.()-]+"
    r"(?:\s*(?:and|,|to)\s*(?:sections?|subsections?|paragraphs?|ss?\.|paras?\.?)?\s*[A-Za-z0-9.()-]+)*"
    r"\s+of\s+the\s+"
    r"|"
    r"(?:pursuant\s+to|under)\s+the\s+"
    r")$"
)

ACT_SUFFIX_RE = re.compile(
    r"^\s*,?\s*(?:S\.C\.|R\.S\.C\.|R\.S\.)\s*\d{4}\s*,?\s*c\.\s*[\w\d.-]+"
)

ACT_RE = re.compile(
    r"(?:(?:pursuant\s+to|under)\s+)?"
    r"(?:"
    r"(?:sections?|subsections?|paragraphs?|ss?\.|paras?\.?)\s*"
    r"[A-Za-z0-9.()-]+"
    r"(?:\s*(?:and|,|to)\s*(?:sections?|subsections?|paragraphs?|ss?\.|paras?\.?)?\s*[A-Za-z0-9.()-]+)*"
    r"\s+(?:of\s+the\s+)?"
    r")?"
    rf"(?:the\s+)?(?:"
    rf"(?!{_ACT_STOP}\s+Act\b){_INSTRUMENT_NAME}\s+Act\b|"
    rf"(?!{_ACT_STOP}\s+Regulations?\b){_INSTRUMENT_NAME}\s+Regulations?\b|"
    r"(?:Canada\s+Labour\s+Code|Labour\s+Relations\s+Code|Labour\s+Code|"
    r"Income\s+Tax\s+Code|Criminal\s+Code|Civil\s+Code)\b|"
    r"(?:Federal\s+Courts?\s+Rules|Tax\s+Court\s+of\s+Canada\s+Rules|"
    r"Rules\s+of\s+Civil\s+Procedure)\b|"
    r"(?:Canadian\s+)?Charter(?:\s+of\s+Rights(?:\s+and\s+Freedoms)?)?\b"
    r")"
    r"(?:\s*,?\s*(?:S\.C\.|R\.S\.C\.|R\.S\.)\s*\d{4}\s*,?\s*c\.\s*[\w\d.-]+)?"
)

_CASE_LEAD_STOP = (
    r"(?!(?:In|See|Also|As|The|And|But|Per|Cf|Following|Applying|Under|From"
    r"|Since|While|Although|Then|Thus|However|Both|Like|Unlike|Accordingly"
    r"|Moreover|Similarly|Indeed|Further|Furthermore|Here|There|This|That"
    r"|These|Those|When|Where|If|Citing|Relying|Contrary|Recently)\s)"
)
_PARTY_WORD = r"(?:[A-ZÀ-Ÿ][A-Za-zÀ-ÿ.'’&\-]*|\((?![^()\n]*\d)[^()\n]{1,80}\))"
_RE_WORD = rf"(?:{_PARTY_WORD}|\d[\w().]*)"
_PARTY_CONN = r"(?:(?:of|and|for|de|du|la|le|des|et|&)(?:\s+the)?)"
_PARTY = rf"{_PARTY_WORD}(?:\s+(?:{_PARTY_CONN}\s+)?{_PARTY_WORD}){{0,9}}"
_SHORT_PARTY = rf"{_PARTY_WORD}(?:\s+(?:{_PARTY_CONN}\s+)?{_PARTY_WORD}){{0,5}}"
_RE_PARTY = rf"{_PARTY_WORD}(?:\s+(?:{_PARTY_CONN}\s+)?{_RE_WORD}){{0,7}}"

CASE_RE = re.compile(
    rf"\b{_CASE_LEAD_STOP}{_PARTY}\s+vs?\.?\s+{_PARTY}"
    rf"|\bReference\s+re\s+{_RE_PARTY}"
    rf"|\b{_CASE_LEAD_STOP}{_SHORT_PARTY},\s+Re\b"
    rf"|\bRe\s+{_RE_PARTY}"
)

PATTERNS = {
    "judge": JUDGE_RE,
    "law": LAW_RE,
    "act": ACT_RE,
    "case": CASE_RE,
}

INDEX_FILENAMES = {
    "judge": "judges.json",
    "law": "laws.json",
    "act": "acts.json",
    "case": "cases.json",
}

_ACT_PREFIX_WINDOW = 160

def _normalize_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s+,", ",", s)

def _normalize_charter(s: str) -> str:
    s = _normalize_ws(s)
    if re.fullmatch(r"Charter", s, re.I):
        return "Canadian Charter of Rights and Freedoms"
    if re.fullmatch(r"Canadian\s+Charter", s, re.I):
        return "Canadian Charter of Rights and Freedoms"
    if re.search(r"Charter\s+of\s+Rights(?:\s+and\s+Freedoms)?$", s, re.I):
        return "Canadian Charter of Rights and Freedoms"
    return s

def _collect_instrument_hits(
    flat: str,
    text: str,
    name_re: re.Pattern,
    *,
    allow_prefix: bool,
    allow_suffix: bool,
    normalize=None,
) -> list[dict]:
    hits: list[dict] = []
    for m in name_re.finditer(flat):
        start, end = m.start(), m.end()
        if allow_prefix:
            left = flat[max(0, start - _ACT_PREFIX_WINDOW) : start]
            pref = ACT_PREFIX_RE.search(left)
            if pref:
                start = max(0, start - _ACT_PREFIX_WINDOW) + pref.start()
        if allow_suffix:
            suf = ACT_SUFFIX_RE.match(flat[end:])
            if suf:
                end = end + suf.end()
        expr = _normalize_ws(flat[start:end])
        if normalize:
            expr = normalize(expr)
        hits.append({"text": expr, "line": text[:start].count("\n") + 1})
    return hits

def extract_acts(flat: str, text: str) -> list[dict]:
    hits: list[dict] = []
    hits.extend(
        _collect_instrument_hits(
            flat, text, ACT_NAME_RE, allow_prefix=True, allow_suffix=True
        )
    )
    hits.extend(
        _collect_instrument_hits(
            flat, text, REGULATION_NAME_RE, allow_prefix=True, allow_suffix=True
        )
    )
    hits.extend(
        _collect_instrument_hits(
            flat, text, CODE_RE, allow_prefix=True, allow_suffix=False
        )
    )
    hits.extend(
        _collect_instrument_hits(
            flat, text, RULES_RE, allow_prefix=True, allow_suffix=False
        )
    )
    hits.extend(
        _collect_instrument_hits(
            flat,
            text,
            CHARTER_RE,
            allow_prefix=True,
            allow_suffix=False,
            normalize=_normalize_charter,
        )
    )
    return hits

_CASE_V_RE = re.compile(r"\s+vs?\.?\s+")
_CASE_RE_SUFFIX_RE = re.compile(r"^(.*),\s+Re$")
_CASE_REFERENCE_RE = re.compile(r"^Reference\s+re\s+")
_CASE_ROLE_RE = re.compile(
    r"\s*\([^()]*\b(?:applicants?|respondents?|appellants?|plaintiffs?|defendants?"
    r"|petitioners?|interveners?|demand\w*|d[ée]fend\w*|intim\w*|supra)\b[^()]*\)",
    re.I,
)
_CASE_COURT_TAIL_RE = re.compile(
    r"\s+(?:Federal Court|Supreme Court|Court of Appeal|Tax Court|Court of Queen's Bench"
    r"|[IVX]+\.\s+[A-Z])\b.*$"
)

def normalize_case(s: str) -> str:
    s = _CASE_ROLE_RE.sub("", _normalize_ws(s))
    s = _CASE_COURT_TAIL_RE.sub("", s)
    s = _CASE_V_RE.sub(" v. ", s, count=1)
    m = _CASE_RE_SUFFIX_RE.match(s)
    if m:
        s = "Re " + m.group(1)
    s = _CASE_REFERENCE_RE.sub("Re ", s)
    return s.rstrip(".,;: ")

def extract_cases(flat: str, text: str) -> list[dict]:
    return [
        {"text": normalize_case(m.group(0)), "line": text[: m.start()].count("\n") + 1}
        for m in CASE_RE.finditer(flat)
    ]

_JUDGE_PARTICLE_RE = re.compile(rf"^{_JUDGE_PARTICLE}$")

def normalize_judge_title(prefix: str | None, name: str) -> str | None:
    tokens: list[str] = []
    for tok in name.split():
        possessive = re.match(r"^(.+?)['’]s?$", tok)
        if possessive:
            tokens.append(possessive.group(1))
            break
        tokens.append(tok)
    tokens = [t for t in tokens if not re.fullmatch(r"[A-ZÀ-Ÿ]\.", t)]
    if not tokens:
        return None
    surname = tokens[-1]
    if len(tokens) >= 2 and _JUDGE_PARTICLE_RE.match(tokens[-2]):
        surname = f"{tokens[-2]} {surname}"
    if prefix and "Chief" in prefix:
        title = "C.J."
    elif prefix and "Deputy" in prefix:
        title = "D.J."
    else:
        title = "J."
    return f"{surname}, {title}"

def _split_judge_names(names: str) -> list[str]:
    parts = re.split(r"\s+(?:and|&)\s+", names)
    surnames: list[str] = []
    for part in parts:
        tokens = part.split()
        if not tokens:
            continue
        surname = tokens[-1]
        if len(tokens) >= 2 and _JUDGE_PARTICLE_RE.match(tokens[-2]):
            surname = f"{tokens[-2]} {surname}"
        surnames.append(surname)
    return surnames

def normalize_judge_suffix(names: str, title: str) -> list[str]:
    title = title.rstrip()
    if title in ("JJ.", "JJ.A."):
        unit = "J.A." if title == "JJ.A." else "J."
        return [f"{surname}, {unit}" for surname in _split_judge_names(names)]
    surnames = _split_judge_names(names)
    if len(surnames) == 1:
        return [f"{surnames[0]}, {title}"]
    return [f"{surname}, {title}" for surname in surnames]

def extract_judges(line: str) -> list[str]:
    found: list[str] = []
    for m in JUDGE_SUFFIX_RE.finditer(line):
        found.extend(normalize_judge_suffix(m.group("names"), m.group("title")))
    for m in JUDGE_TITLE_RE.finditer(line):
        judge = normalize_judge_title(m.group("prefix"), m.group("name"))
        if judge:
            found.append(judge)
    return found

def normalize_law(subject: str, num: str) -> str | None:
    subject = subject.replace("\xad", "").replace("\xa0", " ")
    subject = _normalize_ws(subject).strip(" -–—,;:")
    if len(subject) < 2 or len(subject) > 60:
        return None
    if re.search(r"[()]", subject):
        return None
    if re.search(r"\b(?:ed\.|ibid|supra|http|January|February|March|April|May|June|July|August|September|October|November|December)\b", subject, re.I):
        return None
    if subject.startswith(",") or not subject[0].isupper():
        return None
    if not re.search(r"[A-Za-zÀ-ÿ]{3,}", subject):
        return None
    return f"{subject} - Topic {num}"

def extract_laws(line: str) -> list[str]:
    found: list[str] = []
    for m in LAW_RE.finditer(line):
        law = normalize_law(m.group("subject"), m.group("num"))
        if law:
            found.append(law)
    return found

def extract_from_text(text: str) -> dict[str, list[dict]]:
    matches: dict[str, list[dict]] = {name: [] for name in PATTERNS}

    for line_no, line in enumerate(text.splitlines(), start=1):
        for judge in extract_judges(line):
            matches["judge"].append({"text": judge, "line": line_no})
        for law in extract_laws(line):
            matches["law"].append({"text": law, "line": line_no})

    flat = text.replace("\n", " ")
    matches["act"] = extract_acts(flat, text)
    matches["case"] = extract_cases(flat, text)
    return matches

def extract_from_file(path: Path) -> dict[str, list[dict]]:
    return extract_from_text(path.read_text(encoding="utf-8", errors="replace"))

def load_indexes(
    index_dir: Path,
    types: list[str],
) -> dict[str, dict[str, list[str]]]:
    inverted: dict[str, dict[str, list[str]]] = {}
    for t in types:
        path = index_dir / INDEX_FILENAMES[t]
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} não encontrado. Rode: python regex_extraction/extract_regex.py"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        inverted[t] = data["index"]
    return inverted

def build_inverted_index(
    matches_by_doc: dict[str, dict],
) -> dict[str, dict[str, list[str]]]:
    postings: dict[str, dict[str, set[str]]] = {
        name: defaultdict(set) for name in PATTERNS
    }

    for doc_id, found in matches_by_doc.items():
        for pattern_name in PATTERNS:
            for hit in found.get(pattern_name, []):
                postings[pattern_name][hit["text"]].add(doc_id)

    index: dict[str, dict[str, list[str]]] = {}
    for pattern_name, expr_map in postings.items():
        entries = {
            expr: sorted(docs) for expr, docs in expr_map.items()
        }
        index[pattern_name] = dict(
            sorted(
                entries.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        )
    return index

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extrai juízes, leis, atos e nomes de casos; gera matches e índices invertidos "
            "separados"
        )
    )
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--matches-output", type=Path, default=DEFAULT_MATCHES_OUTPUT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limita o número de arquivos processados",
    )
    args = parser.parse_args()

    if not args.cases_dir.is_dir():
        raise FileNotFoundError(f"Diretório de cases não encontrado: {args.cases_dir}")

    files = sorted(args.cases_dir.glob("*.txt"))
    if args.max_docs is not None:
        files = files[: args.max_docs]

    results: dict[str, dict] = {}
    counts = {name: 0 for name in PATTERNS}

    for i, path in enumerate(files, start=1):
        found = extract_from_file(path)
        if any(found[name] for name in PATTERNS):
            results[path.stem] = found
            for name in PATTERNS:
                counts[name] += len(found[name])
        if i % 1000 == 0:
            print(f"  processados: {i}/{len(files)}")

    inverted = build_inverted_index(results)
    vocab_sizes = {name: len(inverted[name]) for name in PATTERNS}

    matches_payload = {
        "cases_dir": str(args.cases_dir),
        "num_files": len(files),
        "num_files_with_matches": len(results),
        "counts": counts,
        "patterns": {name: pat.pattern for name, pat in PATTERNS.items()},
        "matches": results,
    }

    args.matches_output.parent.mkdir(parents=True, exist_ok=True)
    args.matches_output.write_text(
        json.dumps(matches_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    args.index_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for pattern_name, filename in INDEX_FILENAMES.items():
        out_path = args.index_dir / filename
        payload = {
            "type": pattern_name,
            "pattern": PATTERNS[pattern_name].pattern,
            "num_expressions": vocab_sizes[pattern_name],
            "index": inverted[pattern_name],
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(out_path)

    print(f"Arquivos lidos: {len(files)}")
    print(f"Arquivos com match: {len(results)}")
    print(
        f"Ocorrências — Juízes: {counts['judge']}  |  Leis: {counts['law']}  |  "
        f"Atos: {counts['act']}  |  Casos: {counts['case']}"
    )
    print(
        f"Vocabulário — Juízes: {vocab_sizes['judge']}  |  Leis: {vocab_sizes['law']}  |  "
        f"Atos: {vocab_sizes['act']}  |  Casos: {vocab_sizes['case']}"
    )
    print(f"Matches: {args.matches_output}")
    for path in written:
        print(f"Índice:  {path}")

if __name__ == "__main__":
    main()
