from __future__ import annotations

def rrf_fuse_rankings(
    rankings: list[list[str]],
    *,
    k: int,
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    if not rankings:
        return []
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights deve ter o mesmo tamanho que rankings")

    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        if w == 0.0:
            continue
        for rank, doc_id in enumerate(ranking):
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (rrf_k + rank + 1)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:k]
