"""질의 재작성 → 검색 → 프롬프트 구성 → 응답 생성으로 이어지는 RAG 파이프라인.

프롬프트 문구는 rag.prompts 단일 소스를 사용한다 (여기서 재정의하지 않는다).
"""

from __future__ import annotations

from typing import Dict

from .index import retrieve
from .llm import chat
from .prompts import (  # noqa: F401 — 하위 호환 재노출 (노트북·평가 스크립트)
    PROMPT_BUILDERS,
    SYSTEM_GEN,
    SYSTEM_PROMPTS,
    SYSTEM_REWRITE,
    build_prompt,
    build_prompt_fewshot,
)


def rewrite_query(question: str) -> str:
    user = f"사용자 질문: {question}\n재작성된 검색 질의:"
    return chat(SYSTEM_REWRITE, user, max_tokens=64, temperature=0.0)


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
