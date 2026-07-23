"""질의 재작성 → 검색 → 프롬프트 구성 → 응답 생성으로 이어지는 RAG 파이프라인."""

from __future__ import annotations

from typing import Dict, List

from .index import retrieve
from .llm import chat


def rewrite_query(question: str) -> str:
    system = (
        "당신은 검색 질의 재작성 전문가입니다. 사용자의 모호하거나 추상적인 질문을 "
        "검색엔진에 적합하도록 핵심 개체·키워드 중심의 명확한 한국어 검색 질의로 바꾸세요. "
        "설명 없이 재작성된 질의 한 줄만 출력하세요."
    )
    user = f"사용자 질문: {question}\n재작성된 검색 질의:"
    return chat(system, user, max_tokens=64, temperature=0.0)


# 고정 필수 규칙: ① 참고 문서만을 기반으로 답변 ② 문서에 없으면 찾을 수 없다고만 답변
# (lenient(일반 지식 보완) 모드는 ① 고정 규칙과 충돌해 제거함)
_SYSTEM_GEN = (
    "당신은 한국어로 답하는 QA assistant입니다. "
    "반드시 주어진 참고 문서만을 기반으로 답변하세요. "
    "문서에서 찾을 수 없는 정보는 '제공된 문서에서 찾을 수 없습니다'라고만 답하세요. "
    "답변에 사용한 문서 번호를 [문서 n] 형태로 인용하세요."
)

# strict/fewshot은 시스템 규칙은 동일하고, user 메시지에 few-shot 예시를 넣느냐만 다름
# (도현님 프롬프트 A/B 실험: fewshot이 faithfulness/answer_relevancy 둘 다 우위로 채택됨)
SYSTEM_PROMPTS = {
    "strict": _SYSTEM_GEN,
    "fewshot": _SYSTEM_GEN,
}

# few-shot 예시 2개: ① 다중 문서 인용(근거 아닌 문서는 인용 안 함) ② 근거 없음 → 기권
# prompts.py의 _FEWSHOT과 동일 텍스트(도현님 실험 원본과 일치시켜야 재현 가능)
_FEWSHOT = """다음은 답변 형식 예시입니다.

# 예시 1 (근거가 된 문서를 모두 인용, 근거 아닌 문서는 인용하지 않음)
# 참고 문서
[문서 1] (id=ex-a, title=에베레스트산)
에베레스트산은 해발 8,848m로 지구에서 가장 높은 산이다.
[문서 2] (id=ex-b, title=백두산)
백두산은 한반도에서 가장 높은 산이다.
[문서 3] (id=ex-c, title=에베레스트산 등반사)
에베레스트산은 네팔과 중국 티베트 자치구의 국경에 걸쳐 있다.
# 질문
세계에서 가장 높은 산은 어디에 있고 높이는 얼마인가요?
# 답변
세계에서 가장 높은 산은 에베레스트산으로, 높이는 해발 8,848m입니다 [문서 1]. 네팔과 중국 티베트 자치구의 국경에 걸쳐 있습니다 [문서 3].

# 예시 2 (문서에 근거 없음 → 지어내지 말고 거절)
# 참고 문서
[문서 1] (id=ex-d, title=커피)
커피는 커피나무 열매의 씨앗을 볶아 만든 음료다.
# 질문
녹차에 들어있는 카페인 함량은?
# 답변
제공된 문서에서 찾을 수 없습니다.

이제 아래 실제 질문에 위 형식으로 답하세요.
"""


def build_prompt(question: str, contexts: List[Dict], mode: str = "strict", snippet_len: int = 800):
    if mode not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown mode: {mode!r}, expected one of {list(SYSTEM_PROMPTS)}")
    ctx = "\n\n".join(
        f"[문서 {i+1}] (id={c['corpus_id']}, title={c['title']})\n{c['text'][:snippet_len]}"
        for i, c in enumerate(contexts)
    )
    prefix = _FEWSHOT if mode == "fewshot" else ""
    user = f"{prefix}# 참고 문서\n{ctx}\n\n# 질문\n{question}\n\n# 답변"
    return SYSTEM_PROMPTS[mode], user


def rag_answer(
    question: str,
    collection,
    embed_model,
    query_prefix: str = "",
    k: int = 5,
    mode: str = "strict",
) -> Dict:
    rewritten = rewrite_query(question)
    contexts = retrieve(collection, embed_model, rewritten, query_prefix, k=k)
    system, user = build_prompt(question, contexts, mode)
    answer = chat(system, user, max_tokens=512, temperature=0.0)
    return {
        "question": question,
        "rewritten": rewritten,
        "contexts": contexts,
        "mode": mode,
        "answer": answer,
    }
