Recuperação de casos jurídicos similares.

Dada uma petição/consulta, o sistema devolve os documentos do corpus mais
prováveis de serem relevantes. Há **dois modelos** que depois se **combinam**:

**BM25:** Semelhança de texto (bag-of-words) no corpus limpo
**Regex:** Documentos que citam os mesmos juízes, leis, atos ou casos

## O que cada pasta faz

```
BM25/
├── CorpusColiee/          # dados (cases/, cases_clean/, labels)
├── common/                # caminhos e constantes compartilhadas
├── cleaning/              # 1. trata o texto dos casos
├── lexical_bm25/          # 2. índice e ranking BM25
├── regex_extraction/      # 3. extrai citações e ranqueia por regex
└── hybrid_fusion/         # 4. junta BM25 + regex
```

### 1. Limpeza (`cleaning/`)

O BM25 funciona melhor com texto sem ruído. O script aplica um pipeline fixo
(francês bilíngue, boilerplate, datas/nomes raros, quebras de linha, marcadores)
e grava o resultado em `cases_clean/`.

O **regex usa o corpus bruto** (`cases/`), para não perder citações.

### 2. BM25 (`lexical_bm25/`)

1. `tokenize_corpus.py` — tokeniza `cases_clean/` e gera artefatos em `artifacts/`
2. `query.py` / `evaluate_recall.py` — consulta ou mede recall

Parâmetros: **k1=3**, **b=1**. O ranking final do BM25 já é um **RRF interno**
de quatro listas (passagens, PRF, atos/seções, query completa).

### 3. Regex (`regex_extraction/`)

1. `extract_regex.py` — varre `cases/`, extrai juízes / leis / atos / casos e
   grava índices em `indexes/`
2. `idf_rank.py` — score **wlog** (IDF ponderado por tipo de expressão + bônus de seção em atos)
3. `query.py` — top-k de uma query (combinado ou por tipo de expressão)
4. `evaluate_rank_by_type.py` — Recall@k e NDCG@k **por tipo de expressão** e do combinado

Pesos padrão do combinado: act=1, judge=0.5, law=2, **case=0** (citações de
caso não ajudam neste ranking).

### 4. Fusão (`hybrid_fusion/`)

Recebe o ranking BM25 e o ranking regex e os funde com **RRF**
(pesos padrão BM25=4, regex=1).

- `query.py` — consulta híbrida
- `evaluate_recall.py` — compara BM25-only, regex-only e híbrido

---

## Como rodar

Todos os comandos a partir da pasta `BM25/`:

```bash
pip install -r requirements.txt
```

### Preparar (uma vez)

```bash
python cleaning/case_cleaning.py
python lexical_bm25/tokenize_corpus.py
python regex_extraction/extract_regex.py
```

### Avaliar

```bash
python lexical_bm25/evaluate_recall.py
python regex_extraction/evaluate_rank_by_type.py
python hybrid_fusion/evaluate_recall.py
```

### Consultar um caso

```bash
# BM25
python lexical_bm25/query.py --query-file CorpusColiee\task1_train_files_2026\cases_clean\000002.txt --k 10

# Regex combinado
python regex_extraction/query.py --query-file CorpusColiee\task1_train_files_2026\cases\000002.txt --k 10

# só um tipo de expressão
python regex_extraction/query.py --query-file CorpusColiee\task1_train_files_2026\cases\000002.txt --type act --k 10

# híbrido
python hybrid_fusion/query.py --query-file CorpusColiee\task1_train_files_2026\cases\000002.txt --k 10 --show-sources
```

---

## Arquivos principais

| Arquivo | Função |
|---|---|
| `cleaning/case_cleaning.py` | Limpa o corpus → `cases_clean/` |
| `lexical_bm25/tokenize_corpus.py` | Monta índice BM25 |
| `lexical_bm25/fast_bm25.py` | Implementação BM25 |
| `lexical_bm25/query_ops.py` | Tokenização, passagens, PRF, RRF interno |
| `regex_extraction/extract_regex.py` | Aplica regex e grava índices |
| `regex_extraction/loose_match.py` | Matching frouxo (sobrenome, ato sem seção, etc.) |
| `regex_extraction/idf_rank.py` | Ranking wlog (por tipo ou combinado) |
| `hybrid_fusion/fuse.py` | RRF genérico |
| `hybrid_fusion/rank.py` | Carrega artefatos e funde os dois rankings |

Dados: `CorpusColiee/task1_train_files_2026/` (`cases/`, `cases_clean/`, labels).
