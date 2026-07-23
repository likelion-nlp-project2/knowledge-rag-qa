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
from typing import Iterator, List, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv

from rag.llm import chat as _chat
from schema import ChunkMetadata, PipelineConfig, RAGResponse, RetrievedChunk

load_dotenv()

SEARCH_API_URL = os.getenv("SEARCH_API_URL", "http://localhost:8080").rstrip("/")


# ── ① 질의 재작성 ──
def rewrite_query(question: str) -> str:
    system = ("당신은 검색 질의 재작성 전문가입니다. 사용자의 모호하거나 추상적인 질문을 "
              "검색엔진에 적합하도록 핵심 개체·키워드 중심의 명확한 한국어 검색 질의로 바꾸세요. "
              "설명 없이 재작성된 질의 한 줄만 출력하세요.")
    user = f"사용자 질문: {question}\n재작성된 검색 질의:"
    return _chat(system, user, max_tokens=64, temperature=0.0)


def hyde_rewrite(question: str) -> str:
    """HyDE: 질문에 대한 가상의 답변 문서를 LLM으로 생성해 검색 질의로 쓴다 (rag/generation.py와 동일 방식)."""
    system = ("당신은 위키백과 스타일의 백과사전 문서를 작성하는 어시스턴트입니다. "
              "사용자의 질문에 대한 답을 이미 안다고 가정하고, 그 답이 담긴 위키백과 문단을 "
              "2~4문장으로 작성하세요. 실제로 맞는 답인지 모르더라도 사실처럼 서술하고, "
              "'모르겠다'거나 질문을 되묻지 마세요. 다른 설명 없이 문단만 출력하세요.")
    user = f"질문: {question}\n가상 답변 문단:"
    return _chat(system, user, max_tokens=200, temperature=0.0)


REWRITE_FNS = {"none": None, "keyword": rewrite_query, "hyde": hyde_rewrite}


# ── ④ 프롬프트 생성 ──
# 고정 필수 규칙: ① 참고 문서만을 기반으로 답변 ② 문서에 없으면 찾을 수 없다고만 답변
SYSTEM_GEN = ("당신은 한국어로 답하는 QA assistant입니다. "
              "반드시 주어진 참고 문서만을 기반으로 답변하세요. "
              "문서에서 찾을 수 없는 정보는 '제공된 문서에서 찾을 수 없습니다'라고만 답하세요. "
              "답변에 사용한 문서 번호를 [문서 n] 형태로 인용하세요.")


def _format_ctx(chunks: List[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[문서 {i + 1}] (id={c.metadata.doc_id}, title={c.metadata.title})\n{c.text}"
        for i, c in enumerate(chunks))


def build_prompt(question: str, chunks: List[RetrievedChunk]) -> Tuple[str, str]:
    """A) 베이스라인: 지시만."""
    user = f"# 참고 문서\n{_format_ctx(chunks)}\n\n# 질문\n{question}\n\n# 답변"
    return SYSTEM_GEN, user


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


def build_prompt_fewshot(question: str, chunks: List[RetrievedChunk]) -> Tuple[str, str]:
    """B) Few-shot: 다중 인용 답변 1개 + 근거없음→거절 1개를 예시로 주입."""
    user = (_FEWSHOT
            + f"\n# 참고 문서\n{_format_ctx(chunks)}"
            + f"\n\n# 질문\n{question}\n\n# 답변")
    return SYSTEM_GEN, user


PROMPT_BUILDERS = {"basic": build_prompt, "fewshot": build_prompt_fewshot}


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
        system, user = PROMPT_BUILDERS[cfg.prompt_style](query, filtered)
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

    system, user = build_prompt("영조는 몇 대 왕인가?", chunks[:1])
    assert "[문서 1]" in user and "영조는 몇 대 왕인가?" in user and "조선 영조" in user
    _, user_fs = build_prompt_fewshot("영조는 몇 대 왕인가?", chunks[:1])
    assert "예시 1" in user_fs and user_fs.endswith("# 답변")
    print("self-check OK")
