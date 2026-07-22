"""질의 재작성 → 검색 → 프롬프트 구성 → 응답 생성으로 이어지는 RAG 파이프라인."""

from __future__ import annotations

from typing import Dict, List

from .index import retrieve
from .llm import chat


def rewrite_query(tok, llm, question: str) -> str:
    system = (
        "당신은 검색 질의 재작성 전문가입니다. 사용자의 모호하거나 추상적인 질문을 "
        "검색엔진에 적합하도록 핵심 개체·키워드 중심의 명확한 한국어 검색 질의로 바꾸세요. "
        "설명 없이 재작성된 질의 한 줄만 출력하세요."
    )
    user = f"사용자 질문: {question}\n재작성된 검색 질의:"
    return chat(tok, llm, system, user, max_new_tokens=64, temperature=0.0)


_CITE = "답변에 사용한 문서 번호를 [문서 n] 형태로 인용하세요."

SYSTEM_PROMPTS = {
    # 문서 안에서만 답한다. 근거가 없으면 없다고 밝히고 멈춘다.
    "strict": (
        "당신은 주어진 참고 문서만을 근거로 한국어로 답하는 assistant입니다. "
        "문서에 없는 내용은 지어내거나 추측하지 말고, 근거가 없으면 "
        "'제공된 문서에서 찾을 수 없습니다'라고만 답하세요. " + _CITE
    ),
    # 문서가 부족하면 일반 지식으로 보완하되, 보완한 부분을 답변에서 밝힌다.
    "lenient": (
        "당신은 주어진 참고 문서를 근거로 한국어로 답하는 assistant입니다. "
        "문서가 질문을 충분히 다루지 못하면 당신의 일반 지식으로 보완해 유용한 답을 주되, "
        "그 부분은 문서 밖 정보임을 답변에서 밝히세요. " + _CITE
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
    tok,
    llm,
    query_prefix: str = "",
    k: int = 5,
    mode: str = "strict",
) -> Dict:
    rewritten = rewrite_query(tok, llm, question)
    contexts = retrieve(collection, embed_model, rewritten, query_prefix, k=k)
    system, user = build_prompt(question, contexts, mode)
    answer = chat(tok, llm, system, user, max_new_tokens=512, temperature=0.2)
    return {
        "question": question,
        "rewritten": rewritten,
        "contexts": contexts,
        "mode": mode,
        "answer": answer,
    }
