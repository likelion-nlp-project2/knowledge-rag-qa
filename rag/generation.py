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


def build_prompt(question: str, contexts: List[Dict], snippet_len: int = 800):
    ctx = "\n\n".join(
        f"[문서 {i+1}] (id={c['corpus_id']}, title={c['title']})\n{c['text'][:snippet_len]}"
        for i, c in enumerate(contexts)
    )
    system = (
        "당신은 주어진 참고 문서를 근거로 한국어로 답하는 assistant입니다. "
        "문서에 없는 내용은 지어내지 말고, 근거가 없으면 '제공된 문서에서 찾을 수 없습니다'라고 답하세요. "
        "답변에 사용한 문서 번호를 [문서 n] 형태로 인용하세요."
    )
    user = f"# 참고 문서\n{ctx}\n\n# 질문\n{question}\n\n# 답변"
    return system, user


def rag_answer(
    question: str,
    collection,
    embed_model,
    tok,
    llm,
    query_prefix: str = "",
    k: int = 5,
) -> Dict:
    rewritten = rewrite_query(tok, llm, question)
    contexts = retrieve(collection, embed_model, rewritten, query_prefix, k=k)
    system, user = build_prompt(question, contexts)
    answer = chat(tok, llm, system, user, max_new_tokens=512, temperature=0.2)
    return {
        "question": question,
        "rewritten": rewritten,
        "contexts": contexts,
        "answer": answer,
    }
