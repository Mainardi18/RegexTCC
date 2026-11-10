from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.coliee_utils import (
    DEFAULT_CASES_CLEAN_DIR,
    DEFAULT_CASES_DIR,
    DEFAULT_PASSAGE_STRIDE,
    DEFAULT_PASSAGE_WINDOW,
)
from lexical_bm25.query_ops import build_passage_corpus, tokenize_doc

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "artifacts"

def iter_cases(cases_dir: Path, max_docs: int | None = None):
    files = sorted(cases_dir.glob("*.txt"))
    for i, path in enumerate(files):
        if max_docs is not None and i >= max_docs:
            break
        text = path.read_text(encoding="utf-8", errors="replace")
        yield path.stem, text

def resolve_cases_dir(preferred: Path) -> Path:
    if preferred.is_dir():
        return preferred
    if preferred == DEFAULT_CASES_CLEAN_DIR and DEFAULT_CASES_DIR.is_dir():
        print(
            f"Aviso: {preferred} nao existe; usando {DEFAULT_CASES_DIR}. "
            "Rode antes: python cleaning/case_cleaning.py"
        )
        return DEFAULT_CASES_DIR
    raise FileNotFoundError(f"Diretorio de cases nao encontrado: {preferred}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokeniza cases_clean/ (enriched + stops + passagens)."
    )
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_CLEAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--passage-window", type=int, default=DEFAULT_PASSAGE_WINDOW)
    parser.add_argument("--passage-stride", type=int, default=DEFAULT_PASSAGE_STRIDE)
    args = parser.parse_args()

    args.cases_dir = resolve_cases_dir(args.cases_dir)

    print(f"Lendo cases: {args.cases_dir}")
    doc_ids: list[str] = []
    texts: list[str] = []
    for doc_id, text in iter_cases(args.cases_dir, args.max_docs):
        doc_ids.append(doc_id)
        texts.append(text)
    print(f"Documentos carregados: {len(texts)}")

    t0 = time.perf_counter()
    tokenized_corpus = [tokenize_doc(t) for t in texts]
    tokenize_s = time.perf_counter() - t0
    print(f"Tempo de tokenizacao: {tokenize_s:.4f} s")

    t1 = time.perf_counter()
    passages, p2d = build_passage_corpus(
        tokenized_corpus,
        window=args.passage_window,
        stride=args.passage_stride,
    )
    passage_s = time.perf_counter() - t1
    print(
        f"Passagens: {len(passages)} "
        f"(w={args.passage_window}, s={args.passage_stride}) em {passage_s:.1f}s"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "tokenized_corpus.pkl").open("wb") as f:
        pickle.dump(tokenized_corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (args.out_dir / "doc_ids.json").open("w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
    with (args.out_dir / "passages.pkl").open("wb") as f:
        pickle.dump(passages, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(args.out_dir / "passage_to_doc.npy", p2d)

    meta = {
        "num_docs": len(doc_ids),
        "tokenize_seconds": tokenize_s,
        "cases_dir": str(args.cases_dir),
        "enriched": True,
        "drop_stopwords": True,
        "passages": True,
        "num_passages": len(passages),
        "passage_window": args.passage_window,
        "passage_stride": args.passage_stride,
        "passage_seconds": passage_s,
    }
    with (args.out_dir / "tokenize_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Artefatos salvos em: {args.out_dir}")

if __name__ == "__main__":
    main()
