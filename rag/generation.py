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
    SYSTEM_HYDE,
    SYSTEM_PROMPTS,
    SYSTEM_REWRITE,
    build_prompt,
    build_prompt_fewshot,
)


def rewrite_query(question: str) -> str:
    user = f"사용자 질문: {question}\n재작성된 검색 질의:"
    return chat(SYSTEM_REWRITE, user, max_tokens=64, temperature=0.0)


def hyde_rewrite(question: str) -> str:
    """HyDE: 질문에 대한 가상의 답변 문서를 LLM으로 생성해 검색 질의로 쓴다.

    실제 정답인지는 상관없다 — 질문(짧고 키워드성)보다 '문서처럼 생긴 텍스트'가
    dense 임베딩 공간에서 실제 코퍼스 문서와 더 가깝게 놓인다는 게 핵심 아이디어.
    """
    user = f"질문: {question}\n가상 답변 문단:"
    return chat(SYSTEM_HYDE, user, max_tokens=200, temperature=0.0)


REWRITE_FNS = {"none": None, "keyword": rewrite_query, "hyde": hyde_rewrite}

# mode 하위 호환: 예전 "strict" = 지시형 = PROMPT_BUILDERS["basic"]
_MODE_TO_STYLE = {"strict": "basic", "basic": "basic", "fewshot": "fewshot"}


def rag_answer(
    question: str,
    collection,
    embed_model,
    query_prefix: str = "",
    k: int = 5,
    mode: str = "strict",
    rewrite: str = "keyword",
) -> Dict:
    """rewrite: "none"(원 질의 그대로) | "keyword"(기존 키워드 재작성) | "hyde"(HyDE)."""
    if rewrite not in REWRITE_FNS:
        raise ValueError(f"unknown rewrite: {rewrite!r}, expected one of {list(REWRITE_FNS)}")
    if mode not in _MODE_TO_STYLE:
        raise ValueError(f"unknown mode: {mode!r}, expected one of {list(_MODE_TO_STYLE)}")
    rewrite_fn = REWRITE_FNS[rewrite]
    rewritten = rewrite_fn(question) if rewrite_fn else question
    contexts = retrieve(collection, embed_model, rewritten, query_prefix, k=k)
    system, user = PROMPT_BUILDERS[_MODE_TO_STYLE[mode]](question, contexts, snippet_len=800)
    answer = chat(system, user, max_tokens=512, temperature=0.0)
    return {
        "question": question,
        "rewritten": rewritten,
        "rewrite_mode": rewrite,
        "contexts": contexts,
        "mode": mode,
        "answer": answer,
    }
