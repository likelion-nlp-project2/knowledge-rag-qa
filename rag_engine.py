"""RAG 파이프라인: 검색 API + LLM 생성 연동 드라이버.

- 검색: rag/server.py 가 노출하는 /search API (dense→필터→리랭크→윈도우 근거구절).
  Cloudflare Tunnel 공개 URL 또는 localhost. → SEARCH_API_URL
- 생성: OpenAI chat API, 기본 GPT-4o-mini (rag/llm.py).
  → OPENAI_API_KEY / LLM_API_URL / LLM_MODEL

흐름: ① 질의 재작성 → ② 검색 API → ③ 스레숄드 필터 → ④ 프롬프트 → ⑤ LLM 응답.
"""
from __future__ import annotations

import math
import os
import re
import time
from typing import Iterator, List

import requests
import streamlit as st
from dotenv import load_dotenv

from rag.llm import chat as _chat
from rag.prompts import PROMPT_BUILDERS, SYSTEM_HYDE, SYSTEM_REWRITE
from schema import ChunkMetadata, PipelineConfig, RAGResponse, RetrievedChunk

load_dotenv()

SEARCH_API_URL = os.getenv("SEARCH_API_URL", "http://localhost:8080").rstrip("/")


# ── ① 질의 재작성 (프롬프트: rag.prompts.SYSTEM_REWRITE) ──
def rewrite_query(question: str) -> str:
    user = f"사용자 질문: {question}\n재작성된 검색 질의:"
    return _chat(SYSTEM_REWRITE, user, max_tokens=64, temperature=0.0)


def hyde_rewrite(question: str) -> str:
    """HyDE: 질문에 대한 가상의 답변 문서를 LLM으로 생성해 검색 질의로 쓴다 (rag/generation.py와 동일 방식)."""
    user = f"질문: {question}\n가상 답변 문단:"
    return _chat(SYSTEM_HYDE, user, max_tokens=200, temperature=0.0)


REWRITE_FNS = {"none": None, "keyword": rewrite_query, "hyde": hyde_rewrite}


# ── ④ 프롬프트 입력 변환 (조립은 rag.prompts.PROMPT_BUILDERS) ──
def _to_ctx_dicts(chunks: List[RetrievedChunk]) -> List[dict]:
    """검색 청크 → rag.prompts 가 받는 contexts 형식."""
    return [{"corpus_id": c.metadata.doc_id, "title": c.metadata.title, "text": c.text}
            for c in chunks]


# ── ② 검색 API ──
def _to_chunks(data: dict) -> List[RetrievedChunk]:
    """검색 API 응답(SearchResponse) → RetrievedChunk 목록."""
    chunks = []
    for rank, res in enumerate(data["results"], 1):
        rs, ds = res["rerank_score"], res["dense_score"]
        # ponytail: cross-encoder 로짓을 sigmoid로 0~1 정규화해 스레숄드 슬라이더와 맞춘다.
        # rerank=none이면 rerank_score==dense_score(이미 0~1 근처)라 그대로 쓴다.
        final = ds if data["reranker"] == "none" else 1 / (1 + math.exp(-rs))
        m = res["metadata"]
        chunks.append(RetrievedChunk(
            doc_id=m["doc_id"],
            text=res["text"],
            final_score=round(final, 4),
            rerank_score=round(rs, 4),
            dense_score=round(ds, 4),
            rank=rank,
            metadata=ChunkMetadata(**m),
        ))
    return chunks


def _search_api(query: str, cfg: PipelineConfig) -> List[RetrievedChunk]:
    payload = {"query": query, "k": cfg.top_k_retrieval}
    if not cfg.use_reranker:
        payload["rerank"] = "none"
    r = requests.post(f"{SEARCH_API_URL}/search", json=payload, timeout=60)
    r.raise_for_status()
    return _to_chunks(r.json())


@st.cache_data(ttl=60)
def get_backend_health() -> dict | None:
    """사이드바 시스템 정보용 /health. 연결 안 되면 None."""
    try:
        return requests.get(f"{SEARCH_API_URL}/health", timeout=5).json()
    except requests.RequestException:
        return None


def stream_tokens(text: str, delay: float = 0.02) -> Iterator[str]:
    """`st.write_stream`용 토큰 제너레이터 — 완성된 답변을 스트리밍처럼 출력."""
    for tok in re.findall(r"\S+\s*", text):
        yield tok
        time.sleep(delay)


# ── 파이프라인 진입점 ──
def run_pipeline(query: str, cfg: PipelineConfig) -> RAGResponse:
    """UI는 이 함수 하나만 호출한다."""
    times = {}

    t0 = time.perf_counter()
    rewrite_fn = REWRITE_FNS.get(cfg.rewrite_mode)
    try:
        rq = rewrite_fn(query) if rewrite_fn else None
    except requests.RequestException:
        rq = None   # LLM이 죽어 있어도 검색까지는 보여준다
    times["rewrite"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    retrieved = _search_api(rq or query, cfg)
    times["retrieval"] = round(time.perf_counter() - t0, 3)

    filtered = [c for c in retrieved if c.final_score >= cfg.min_score][: cfg.top_k_context]

    answer, times["llm"] = "", 0.0
    if filtered:
        system, user = PROMPT_BUILDERS[cfg.prompt_style](query, _to_ctx_dicts(filtered))
        t0 = time.perf_counter()
        answer = _chat(system, user, max_tokens=512, temperature=0.0)
        times["llm"] = round(time.perf_counter() - t0, 3)

    return RAGResponse(
        query=query,
        rewritten_query=rq,
        generated_answer=answer,
        retrieved_chunks=retrieved,
        filtered_chunks=filtered,
        execution_time_sec=times,
    )


if __name__ == "__main__":
    # 셀프 체크(네트워크 불필요): 응답 매핑·정규화·프롬프트 조립 계약 검증
    meta = {"doc_id": "11098", "title": "조선 영조", "url": "u", "doc_type": "일반",
            "categories": "역사", "matched_chunks": "11098#0", "window_chunks": "11098#0|11098#1"}
    data = {"query": "q", "reranker": "bge-reranker-v2-m3", "results": [
        {"text": "영조는 조선의 제21대 왕이다.", "rerank_score": 2.0, "dense_score": 0.8, "metadata": meta},
        {"text": "관련 낮은 문단.", "rerank_score": -3.0, "dense_score": 0.4, "metadata": dict(meta, doc_id="99")},
    ]}
    chunks = _to_chunks(data)
    assert [c.rank for c in chunks] == [1, 2]
    assert chunks[0].final_score > 0.55 > chunks[1].final_score, "sigmoid 정규화가 스레숄드와 맞아야 함"
    none_chunks = _to_chunks(dict(data, reranker="none"))
    assert none_chunks[0].final_score == 0.8, "rerank=none이면 dense 점수 그대로"

    ctxs = _to_ctx_dicts(chunks[:1])
    system, user = PROMPT_BUILDERS["basic"]("영조는 몇 대 왕인가?", ctxs)
    assert "[문서 1]" in user and "영조는 몇 대 왕인가?" in user and "조선 영조" in user
    _, user_fs = PROMPT_BUILDERS["fewshot"]("영조는 몇 대 왕인가?", ctxs)
    assert "예시 1" in user_fs and user_fs.endswith("# 답변")
    print("self-check OK")
