"""Recall/MRR/nDCG/Hit@k 평가.

retrieve_fn(qid, k) -> 랭킹된 corpus_id 리스트 만 넘기면 되므로, 임베딩/컬렉션 구현과
무관하게 순수 함수로 테스트할 수 있다.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import numpy as np

METRIC_NAMES = ("Recall", "MRR", "nDCG", "Hit")


def dcg(rels: Iterable[int]) -> float:
    return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))


def evaluate(
    retrieve_fn: Callable[[str, int], List[str]],
    eval_qids: Iterable[str],
    gold: Dict[str, Dict[str, float]],
    k_list: List[int] = (1, 5, 10),
    top_k: int = 10,
) -> Dict[str, float]:
    acc = {f"{m}@{k}": [] for k in k_list for m in METRIC_NAMES}
    for qid in eval_qids:
        ranked = retrieve_fn(qid, top_k)
        rel = {c for c, s in gold.get(qid, {}).items() if s > 0}
        if not rel:
            continue
        for k in k_list:
            hits = [1 if c in rel else 0 for c in ranked[:k]]
            acc[f"Recall@{k}"].append(sum(hits) / len(rel))
            acc[f"Hit@{k}"].append(1.0 if any(hits) else 0.0)
            rr = next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
            acc[f"MRR@{k}"].append(rr)
            idcg = dcg([1] * min(len(rel), k))
            acc[f"nDCG@{k}"].append(dcg(hits) / idcg if idcg > 0 else 0.0)

    order = [f"{m}@{k}" for k in k_list for m in METRIC_NAMES]
    return {m: float(np.mean(acc[m])) if acc[m] else 0.0 for m in order}


def _self_check():
    gold = {"q1": {"a": 1.0, "b": 1.0}, "q2": {"c": 1.0}}

    def perfect_retrieve(qid, k):
        return list(gold[qid].keys())[:k]

    result = evaluate(perfect_retrieve, ["q1", "q2"], gold, k_list=[1, 2], top_k=2)
    assert result["Hit@1"] == 1.0
    assert result["Recall@2"] == 1.0
    assert result["MRR@1"] == 1.0

    def empty_retrieve(qid, k):
        return []

    zero = evaluate(empty_retrieve, ["q1"], gold, k_list=[1], top_k=1)
    assert zero["Recall@1"] == 0.0 and zero["Hit@1"] == 0.0

    print("self-check ok")


if __name__ == "__main__":
    _self_check()
