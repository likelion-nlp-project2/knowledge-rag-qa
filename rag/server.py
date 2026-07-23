"""검색 API 서버 (FastAPI).

bge-m3 임베더를 한 번만 올리고, 영속 Chroma 컬렉션에서 검색한다.
검색 파이프라인(rag.search): dense N → 필터 → 리랭커 → top-K → 윈도우 근거구절.
Cloudflare Tunnel이 이 서버(http://api:8080)를 공개 URL로 노출한다.

  uvicorn rag.server:app --host 0.0.0.0 --port 8080

엔드포인트:
  GET  /health              상태 + 적재된 청크 수 + 로드된 리랭커
  POST /search              {"query": "...", "k": 5, "rerank": "bge-reranker-v2-m3"}
  GET  /search?q=...&k=5&rerank=none

임베딩 모델은 EMBED_MODEL, 기본 리랭커는 RERANK_MODEL 로 고른다. 리랭커는
요청마다 rerank 파라미터로 바꿀 수 있어(none 포함) 여러 방식을 나란히 비교할 수
있다(처음 쓰는 리랭커는 최초 요청 때 로드되어 캐시된다).
"""

from __future__ import annotations

from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import INFRA, collection_name, get_embed_model, get_rerank_model
from .embedding import load_embedder
from .index import connect, get_or_create_collection
from .rerank import Reranker, load_reranker
from .search import search

app = FastAPI(title="Ko-miracl 검색 API", version="2.1")

_state: dict = {}
_rerankers: dict = {}   # key -> Reranker (지연 로드 캐시)


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = None          # 리랭크 후 상위 청크 수 (기본 INFRA.rerank_top_k)
    n: Optional[int] = None          # dense 후보 수 (기본 INFRA.retrieve_n)
    rerank: Optional[str] = None     # 리랭커 key (기본 INFRA.rerank_model)


class ResultMeta(BaseModel):
    doc_id: str
    title: str
    url: str
    doc_type: str = ""
    categories: str = ""
    matched_chunks: str              # 리랭크 상위로 매칭된 청크 id, '|' join
    window_chunks: str               # 실제 실린 청크(매칭+이웃), '|' join


class SearchResult(BaseModel):
    text: str                        # 근거구절(매칭 청크 + 윈도우) — 전체문서 아님
    rerank_score: float              # 정렬 기준
    dense_score: float               # 참고용(dense 유사도)
    metadata: ResultMeta


class SearchResponse(BaseModel):
    query: str
    reranker: str                    # 이번 검색에 쓴 리랭커 key (비교 추적용)
    results: List[SearchResult]


def _get_reranker(key: str) -> Reranker:
    """리랭커를 지연 로드해 캐시한다."""
    get_rerank_model(key)   # 미등록 key 면 여기서 KeyError
    if key not in _rerankers:
        _rerankers[key] = load_reranker(key, _state["device"], fp16=INFRA.fp16)
        print(f"[server] reranker loaded: {key}")
    return _rerankers[key]


@app.on_event("startup")
def _startup() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = get_embed_model(INFRA.embed_model)
    model = load_embedder(mcfg.hf_id, device, mcfg.max_seq_len, fp16=INFRA.fp16)
    client = connect(INFRA.chroma_host, INFRA.chroma_port)
    collection = get_or_create_collection(client, collection_name(mcfg.key))
    _state.update(model=model, mcfg=mcfg, collection=collection, device=device)
    print(f"[server] {mcfg.hf_id} on {device}, "
          f"collection={collection_name(mcfg.key)} ({collection.count():,} chunks)")
    # 기본 리랭커는 미리 로드(none 이면 즉시). 실패해도 서버는 뜬다.
    try:
        _get_reranker(INFRA.rerank_model)
    except Exception as e:  # noqa: BLE001
        print(f"[server] 기본 리랭커 로드 실패({INFRA.rerank_model}): {e}")


@app.get("/health")
def health() -> dict:
    col = _state.get("collection")
    return {
        "status": "ok" if col is not None else "starting",
        "model": _state.get("mcfg").key if _state.get("mcfg") else None,
        "device": _state.get("device"),
        "chunks": col.count() if col is not None else 0,
        "default_reranker": INFRA.rerank_model,
        "loaded_rerankers": list(_rerankers),
    }


def _search(query: str, k: Optional[int], n: Optional[int],
            rerank: Optional[str]) -> SearchResponse:
    if not query or not query.strip():
        raise HTTPException(400, "query 가 비었습니다")
    if "collection" not in _state:
        raise HTTPException(503, "서버 초기화 중입니다")
    rk = rerank or INFRA.rerank_model
    try:
        reranker = _get_reranker(rk)
    except KeyError as e:
        raise HTTPException(400, str(e))
    mcfg = _state["mcfg"]
    results = search(
        _state["collection"], _state["model"], reranker, query,
        mcfg.query_prefix,
        n=n or INFRA.retrieve_n,
        top_k=k or INFRA.rerank_top_k,
        window=INFRA.context_window,
        per_doc_max=INFRA.per_doc_max_chunks,
        total_max=INFRA.context_max_chunks,
    )
    return SearchResponse(
        query=query, reranker=reranker.key,
        results=[SearchResult(**r) for r in results],
    )


@app.post("/search", response_model=SearchResponse)
def search_post(req: SearchRequest) -> SearchResponse:
    return _search(req.query, req.k, req.n, req.rerank)


@app.get("/search", response_model=SearchResponse)
def search_get(q: str, k: Optional[int] = None, n: Optional[int] = None,
               rerank: Optional[str] = None) -> SearchResponse:
    return _search(q, k, n, rerank)
