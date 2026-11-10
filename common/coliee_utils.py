from __future__ import annotations

from pathlib import Path

BM25_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = BM25_ROOT / "CorpusColiee" / "task1_train_files_2026"
DEFAULT_CASES_DIR = CORPUS_DIR / "cases"
DEFAULT_CASES_CLEAN_DIR = CORPUS_DIR / "cases_clean"
DEFAULT_LABELS = CORPUS_DIR / "clean_task1_train_labels_2026.json"

DEFAULT_BM25_K1 = 3.0
DEFAULT_BM25_B = 1.0
DEFAULT_MAX_QUERY_TOKENS = 300
DEFAULT_PRF_DOCS = 3
DEFAULT_PRF_TERMS = 50
DEFAULT_PASSAGE_WINDOW = 300
DEFAULT_PASSAGE_STRIDE = 200
DEFAULT_RRF_K = 60

def normalize_doc_id(doc_id: str) -> str:
    name = Path(doc_id).name
    return name[:-4] if name.lower().endswith(".txt") else name

def load_query_text(
    query: str | None,
    query_file: Path | None,
    max_chars: int | None = None,
) -> tuple[str, str | None]:
    if query_file is not None:
        path = Path(query_file)
        if not path.is_file():
            raise FileNotFoundError("Arquivo nao encontrado")
        text = path.read_text(encoding="utf-8", errors="replace")
        doc_id = path.stem
    elif query is not None:
        text = query
        doc_id = None
    else:
        raise ValueError("Informe --query ou --query-file.")

    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]

    return text, doc_id

def exclude_doc(
    hits: list[tuple[str, float]],
    exclude_id: str | None,
    k: int,
) -> list[tuple[str, float]]:
    if exclude_id is None:
        return hits[:k]
    excl = normalize_doc_id(exclude_id)
    filtered = [(d, s) for d, s in hits if normalize_doc_id(d) != excl]
    return filtered[:k]
