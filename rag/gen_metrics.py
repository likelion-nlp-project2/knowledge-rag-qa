"""생성(RAG 답변) 규칙 기반 평가 지표 — 계획서 4.3 '생성 주 지표'.

검색 지표(rag/metrics.py)와 분리해, 생성 답변이 '약속한 근거 규칙'을 지켰는지
규칙(정규식 + answerable 라벨)만으로 결정적으로 측정한다. 판정 LLM을 쓰지 않으므로
seed에 무관하게 재현 가능하다(신뢰성 요건). RAGAS(보조 지표)와 역할이 다르다.

측정하는 3가지:
- 인용 정확성: 답변의 [문서 n]이 실제 제시된 검색 문서 범위(1..n_contexts) 안인가
- 환각율: 근거 없는(unanswerable) 질문에 기권하지 않고 답을 지어낸 비율
- 기권 적절성: 답할 수 있으면 답하고(과잉 기권 X), 근거 없으면 기권했는가(양방향)

records 각 원소 형식:
    {"answer": str, "n_contexts": int, "answerable": bool}
  - answerable=True  : dev 정답 쿼리(정답 문서가 인덱스에 있음) → 기권하면 안 됨
  - answerable=False : 무근거 대조군(정답 문서를 인덱스에서 뺀 케이스 A, 또는 코퍼스 밖 질문 B)
                       → 기권해야 정답
"""

from __future__ import annotations

import re
from typing import Dict, List

# strict 프롬프트가 근거 부재 시 내도록 지시하는 문구(rag/generation.py와 일치)
ABSTENTION_MARKER = "제공된 문서에서 찾을 수 없습니다"
_CITE_RE = re.compile(r"\[문서\s*(\d+)\]")


def is_abstention(answer: str) -> bool:
    return ABSTENTION_MARKER in answer


def extract_citations(answer: str) -> List[int]:
    """답변에서 [문서 n] 형태의 인용 번호를 모두 뽑는다."""
    return [int(n) for n in _CITE_RE.findall(answer)]


def citation_accuracy(records: List[Dict]) -> Dict[str, float]:
    """인용 정확성 — 비기권 답변만 대상.

    - citation_validity : 전체 [문서 n] 인용 중 유효 범위(1..n_contexts) 안인 비율
    - citation_coverage : 비기권 답변 중 인용을 1개 이상 단 비율
    """
    total_cites = valid_cites = 0
    answered = cited = 0
    for r in records:
        if is_abstention(r["answer"]):
            continue
        answered += 1
        cites = extract_citations(r["answer"])
        if cites:
            cited += 1
        for n in cites:
            total_cites += 1
            if 1 <= n <= r["n_contexts"]:
                valid_cites += 1
    return {
        "citation_validity": valid_cites / total_cites if total_cites else 0.0,
        "citation_coverage": cited / answered if answered else 0.0,
        "n_answered": float(answered),
    }


def abstention_metrics(records: List[Dict]) -> Dict[str, float]:
    """기권 적절성(양방향) + 환각율.

    answerable=True 인데 기권  → 과잉 기권(over-abstention, 나쁨)
    answerable=False 인데 답함 → 환각(hallucination, 나쁨)
    answerable=False 이고 기권 → 올바른 기권(정답)
    """
    ans = [r for r in records if r["answerable"]]
    unans = [r for r in records if not r["answerable"]]

    over = sum(is_abstention(r["answer"]) for r in ans)               # 답 있는데 기권
    correct_abstain = sum(is_abstention(r["answer"]) for r in unans)  # 근거 없어 기권(정답)
    hallucinated = len(unans) - correct_abstain                       # 근거 없는데 답함(환각)

    n_ans, n_unans = len(ans), len(unans)
    n_correct = (n_ans - over) + correct_abstain
    n_total = n_ans + n_unans

    return {
        # 기권 적절성
        "over_abstention_rate": over / n_ans if n_ans else 0.0,                 # ↓ 좋음
        "correct_abstention_rate": correct_abstain / n_unans if n_unans else 0.0,  # ↑ 좋음
        "abstention_accuracy": n_correct / n_total if n_total else 0.0,         # ↑ 좋음
        # 환각율(= unanswerable 에서 기권 실패 비율)
        "hallucination_rate": hallucinated / n_unans if n_unans else 0.0,       # ↓ 좋음
        "n_answerable": float(n_ans),
        "n_unanswerable": float(n_unans),
    }


def evaluate_generation(records: List[Dict]) -> Dict[str, float]:
    """규칙 기반 생성 주 지표 일괄 계산(인용 정확성 + 기권 적절성 + 환각율)."""
    out: Dict[str, float] = {}
    out.update(citation_accuracy(records))
    out.update(abstention_metrics(records))
    return out


def _self_check():
    records = [
        # 답 있는 질문에 문서1 인용하며 정상 응답 → 이상적
        {"answer": "정답입니다. [문서 1]", "n_contexts": 5, "answerable": True},
        # 답 있는 질문인데 잘못 기권 → 과잉 기권
        {"answer": ABSTENTION_MARKER, "n_contexts": 5, "answerable": True},
        # 근거 없는 질문에 올바르게 기권 → 정답
        {"answer": ABSTENTION_MARKER, "n_contexts": 5, "answerable": False},
        # 근거 없는 질문에 지어냄 → 환각
        {"answer": "아인슈타인입니다. [문서 9]", "n_contexts": 5, "answerable": False},
    ]
    m = evaluate_generation(records)

    assert extract_citations("[문서 1] 그리고 [문서 12]") == [1, 12]
    assert is_abstention("어쩌고 " + ABSTENTION_MARKER)
    # 유효 인용: [문서1](답1), [문서9](환각답, 범위밖) → 2개 중 1개 유효 = 0.5
    assert abs(m["citation_validity"] - 0.5) < 1e-9, m["citation_validity"]
    # 과잉 기권: 답 있는 2개 중 1개 기권 = 0.5
    assert abs(m["over_abstention_rate"] - 0.5) < 1e-9
    # 환각: 근거없는 2개 중 1개 답함 = 0.5
    assert abs(m["hallucination_rate"] - 0.5) < 1e-9
    # 올바른 기권: 근거없는 2개 중 1개 기권 = 0.5
    assert abs(m["correct_abstention_rate"] - 0.5) < 1e-9
    print("gen_metrics self-check ok:", {k: round(v, 3) for k, v in m.items()})


if __name__ == "__main__":
    _self_check()
