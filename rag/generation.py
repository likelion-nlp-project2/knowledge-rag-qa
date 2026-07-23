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
SYSTEM_PROMPTS = {
    "strict": (
        "당신은 한국어로 답하는 QA assistant입니다. "
        "반드시 주어진 참고 문서만을 기반으로 답변하세요. "
        "문서에서 찾을 수 없는 정보는 '제공된 문서에서 찾을 수 없습니다'라고만 답하세요. "
        "답변에 사용한 문서 번호를 [문서 n] 형태로 인용하세요."
    ),
}


def build_prompt(question: str, contexts: List[Dict], mode: str = "strict", snippet_len: int = 800):
    if mode not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown mode: {mode!r}, expected one of {list(SYSTEM_PROMPTS)}")
    ctx = "\n\n".join(
        f"[문서 {i+1}] (id={c['corpus_id']}, title={c['title']})\n{c['text'][:snippet_len]}"
        for i, c in enumerate(contexts)
    )
    user = f"# 참고 문서\n{ctx}\n\n# 질문\n{question}\n\n# 답변"
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
