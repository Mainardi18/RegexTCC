from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.coliee_utils import DEFAULT_CASES_CLEAN_DIR, DEFAULT_CASES_DIR

FRAG = "<FRAGMENT_SUPPRESSED>"
PARA_LINE_RE = re.compile(r"^\s*\[\d+\]\s*$")
_SPACES_RE = re.compile(r"[ \t\xa0]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

DEFAULT_PIPELINE = ("E1", "E2", "E3", "E4", "E5")
DEFAULT_OUT_DIR = DEFAULT_CASES_CLEAN_DIR

def bare(line: str) -> str:
    return _SPACES_RE.sub(" ", line.replace(FRAG, " ")).strip()

def is_para_line(line: str) -> bool:
    return bool(PARA_LINE_RE.match(line))

_INLINE_MARKERS_RE = re.compile(
    r"<FRAGMENT_SUPPRESSED>|\[\d+\]|\[End of document\]|\[Translation\]", re.I
)
_SYMBOLS_ONLY_RE = re.compile(r"^[\W_]+$")

def clean_markers(text: str) -> str:
    text = _INLINE_MARKERS_RE.sub(" ", text).replace("\ufffd", "")
    out: list[str] = []
    for line in text.split("\n"):
        s = _SPACES_RE.sub(" ", line).strip()
        if not s or _SYMBOLS_ONLY_RE.match(s):
            if out and out[-1] != "":
                out.append("")
            continue
        out.append(s)
    return "\n".join(out).strip()

_JOIN_START_RE = re.compile(r"^(?:[,.;:)\]]|[a-zà-ÿ])")
_TRAILING_JOIN_RE = re.compile(
    r"(?:\(|\b(?:of|the|to|under|in|for|by|with|de|du|des|la|le|les))$", re.I
)
_TITLE_TAILS = {
    "Act",
    "Code",
    "Regulations",
    "Rules",
    "Charter",
    "Convention",
    "Order",
    "Protocol",
    "Treaty",
    "Agreement",
}

def join_broken_lines(text: str) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        b = bare(line)
        if out and b and not is_para_line(line):
            prev_raw = out[-1]
            prev = bare(prev_raw)
            if prev and not is_para_line(prev_raw) and (
                _JOIN_START_RE.match(b)
                or _TRAILING_JOIN_RE.search(prev)
                or (
                    b.rstrip(",.;:)") in _TITLE_TAILS
                    and not prev.endswith((".", "?", "!", ":"))
                )
            ):
                out[-1] = prev_raw.rstrip() + " " + line.strip()
                continue
        out.append(line)
    return "\n".join(out)

_FR_WORDS = frozenset(
    "le la les des du de et un une est dans pour qui que sur au aux par ne pas "
    "sont être cette ce ces ses son sa il elle ils leur leurs en ou où avec été "
    "lui mais peut doit tout toute tous selon lorsque présente loi cour "
    "demandeur défendeur demanderesse ministre ainsi alinéa paragraphe article "
    "partie dépens règles délai décision tribunal".split()
)
_EN_WORDS = frozenset(
    "the of and to in is that for on with by was be as it this which or are an "
    "not from at has have had shall may any who would his her their its".split()
)
_WORD_RE = re.compile(r"[a-zà-ÿ]+")
_ELISION_RE = re.compile(r"\b(?:l|d|qu|n|s|j|c)['’](?=[a-zà-ÿ])")
_ACCENT_RE = re.compile(r"[éèêàùçôîâœë]")

def _french_signal(line: str) -> tuple[int, int, int]:
    low = bare(line).lower()
    words = _WORD_RE.findall(low)
    fr = sum(1 for w in words if w in _FR_WORDS) + len(_ELISION_RE.findall(low))
    en = sum(1 for w in words if w in _EN_WORDS)
    return fr, en, len(_ACCENT_RE.findall(low))

def _is_french(fr: int, en: int, accents: int) -> bool:
    if fr >= 2 and fr >= 2 * en:
        return True
    return fr >= 1 and en == 0 and accents >= 1

def remove_french(text: str) -> str:
    lines = text.split("\n")
    signals = [_french_signal(line) for line in lines]
    french = [_is_french(*s) for s in signals]
    nonblank = [i for i, line in enumerate(lines) if bare(line)]
    for pos, i in enumerate(nonblank):
        fr, en, _ = signals[i]
        if french[i] or fr or en:
            continue
        prev_fr = pos > 0 and french[nonblank[pos - 1]]
        next_fr = pos + 1 < len(nonblank) and french[nonblank[pos + 1]]
        if prev_fr and next_fr:
            french[i] = True
    return "\n".join(line for line, is_fr in zip(lines, french) if not is_fr)

def _sentence_key(sent: str) -> str:
    return re.sub(r"\d", "0", re.sub(r"\s+", " ", bare(sent).lower())).strip()

def remove_boilerplate(
    texts: list[str],
    *,
    min_doc_frac: float = 0.01,
    min_words: int = 8,
) -> tuple[list[str], dict]:
    df: Counter[str] = Counter()
    for text in texts:
        keys = set()
        for line in text.split("\n"):
            for sent in _SENT_SPLIT_RE.split(line):
                key = _sentence_key(sent)
                if len(key.split()) >= min_words:
                    keys.add(key)
        df.update(keys)
    threshold = max(2, int(round(min_doc_frac * len(texts))))
    boiler = {k for k, n in df.items() if n >= threshold}
    removed: Counter[str] = Counter()
    out: list[str] = []
    for text in texts:
        new_lines = []
        for line in text.split("\n"):
            sents = _SENT_SPLIT_RE.split(line)
            kept = []
            for sent in sents:
                key = _sentence_key(sent)
                if key in boiler:
                    removed[key] += 1
                else:
                    kept.append(sent)
            if len(kept) == len(sents):
                new_lines.append(line)
            elif kept:
                new_lines.append(" ".join(kept))
        out.append("\n".join(new_lines))
    info = {
        "min_doc_frac": min_doc_frac,
        "doc_threshold": threshold,
        "num_boilerplate_sentences": len(boiler),
        "top_removed": [
            {"sentence": k, "df": df[k], "occurrences": n}
            for k, n in removed.most_common(50)
        ],
    }
    return out, info

_MONTHS = (
    r"January|February|March|April|May|June|July|August|September|October"
    r"|November|December|Jan\.|Feb\.|Mar\.|Apr\.|Jun\.|Jul\.|Aug\.|Sept?\."
    r"|Oct\.|Nov\.|Dec\."
)
_DATE_RE = re.compile(
    rf"(?:\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS}),?\s+\d{{4}}"
    rf"|\b(?:{_MONTHS})\s+\d{{4}}"
    r"|\b\d{4}-\d{2}-\d{2}|\b\d{1,2}/\d{1,2}/\d{2,4})\b"
)
_CITIES_RE = re.compile(
    r"\b(?:Toronto|Montreal|Montréal|Ottawa|Vancouver|Calgary|Edmonton|Winnipeg"
    r"|Halifax|Regina|Saskatoon|Hamilton|Mississauga|Brampton|Qu[eé]bec City"
    r"|Fredericton|Moncton|Charlottetown|Kitchener|Windsor|Burnaby|Laval"
    r"|Gatineau|St\. John's)\b"
)
_CAP_WORD_RE = re.compile(r"\b[A-ZÀ-Ý][a-zà-ÿ'’\-]{2,}\b")
_LOWER_WORD_RE = re.compile(r"\b[a-zà-ÿ][a-zà-ÿ'’\-]{2,}\b")

def remove_specific_facts(
    texts: list[str],
    *,
    max_df: int = 3,
) -> tuple[list[str], dict]:
    lower_vocab: set[str] = set()
    cap_df: Counter[str] = Counter()
    for text in texts:
        lower_vocab.update(_LOWER_WORD_RE.findall(text))
        cap_df.update(set(_CAP_WORD_RE.findall(text)))
    rare = {
        w for w, n in cap_df.items() if n <= max_df and w.lower() not in lower_vocab
    }

    def drop_rare(m: re.Match) -> str:
        return " " if m.group(0) in rare else m.group(0)

    out = []
    for text in texts:
        text = _DATE_RE.sub(" ", text)
        text = _CITIES_RE.sub(" ", text)
        out.append(_CAP_WORD_RE.sub(drop_rare, text))
    info = {
        "max_df": max_df,
        "num_rare_proper_nouns": len(rare),
        "sample": sorted(rare)[:100],
    }
    return out, info

def clean_corpus(texts: list[str]) -> tuple[list[str], dict]:
    infos: dict = {}
    texts = [remove_french(t) for t in texts]
    texts, infos["E2"] = remove_boilerplate(texts)
    texts, infos["E3"] = remove_specific_facts(texts)
    texts = [join_broken_lines(t) for t in texts]
    texts = [clean_markers(t) for t in texts]
    return texts, infos

def clean_text(text: str) -> str:
    text = remove_french(text)
    text = join_broken_lines(text)
    return clean_markers(text)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Pasta de saída dos .txt limpos (padrão: cases_clean/).",
    )
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    if not args.cases_dir.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {args.cases_dir}")

    files = sorted(args.cases_dir.glob("*.txt"))
    if args.max_docs is not None:
        files = files[: args.max_docs]
    if not files:
        raise FileNotFoundError(f"Nenhum .txt em {args.cases_dir}")

    print(f"Lendo {len(files)} docs de {args.cases_dir} ...")
    t0 = time.perf_counter()
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in files]
    cleaned, infos = clean_corpus(texts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path, text in zip(files, cleaned):
        (args.out_dir / path.name).write_text(text, encoding="utf-8")

    boiler_n = infos.get("E2", {}).get("num_boilerplate_sentences", 0)
    rare_n = infos.get("E3", {}).get("num_rare_proper_nouns", 0)
    print(f"Pipeline: {' -> '.join(DEFAULT_PIPELINE)}")
    print(f"Boilerplate removido (frases únicas): {boiler_n}")
    print(f"Nomes próprios raros removidos: {rare_n}")
    print(f"Gravados {len(files)} arquivos em {args.out_dir} ({time.perf_counter() - t0:.1f}s)")

if __name__ == "__main__":
    main()
