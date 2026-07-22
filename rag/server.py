"""검색 API 서버 (FastAPI).

bge-m3 임베더를 한 번만 올리고, 영속 Chroma 컬렉션에서 검색한다.
Cloudflare Tunnel이 이 서버(http://api:8080)를 공개 URL로 노출한다.

  uvicorn rag.server:app --host 0.0.0.0 --port 8080

엔드포인트:
  GET  /health          상태 + 적재된 청크 수
  POST /search          {"query": "...", "k": 5}
  GET  /search?q=...&k=5

임베딩 모델은 환경변수 EMBED_MODEL 로 고른다(기본 bge-m3). 모델을 비교하려면
서로 다른 EMBED_MODEL 로 컬렉션을 각각 적재한 뒤(rag.ingest), 이 서버를 해당
모델로 띄우면 된다.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import INFRA, collection_name, get_embed_model
from .embedding import load_embedder
from .index import connect, fetch_full_documents, get_or_create_collection, retrieve

app = FastAPI(title="Ko-miracl 검색 API", version="2.0")

_state: dict = {}


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class DocMeta(BaseModel):
    doc_id: str
    title: str
    url: str
    doc_type: str = ""
    categories: str = ""
    matched_chunks: str        # 매칭된 청크 id들, '|' join
    n_chunks: int              # 문서 전체 청크 수


class DocResult(BaseModel):
    text: str                  # 문서 전체(모든 청크 이어붙임)
    score: float               # 매칭 청크 최고 유사도
    metadata: DocMeta


class SearchResponse(BaseModel):
    query: str
    documents: List[DocResult]


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


@app.get("/health")
def health() -> dict:
    col = _state.get("collection")
    return {
        "status": "ok" if col is not None else "starting",
        "model": _state.get("mcfg").key if _state.get("mcfg") else None,
        "device": _state.get("device"),
        "chunks": col.count() if col is not None else 0,
    }


def _search(query: str, k: int) -> SearchResponse:
    if not query or not query.strip():
        raise HTTPException(400, "query 가 비었습니다")
    if "collection" not in _state:
        raise HTTPException(503, "서버 초기화 중입니다")
    mcfg = _state["mcfg"]
    collection = _state["collection"]
    # 1) k개 청크 검색 → 2) doc_id 취합해 각 문서 전체 청크 복원
    hits = retrieve(collection, _state["model"], query, mcfg.query_prefix, k=k)
    docs = fetch_full_documents(collection, hits)
    return SearchResponse(
        query=query,
        documents=[DocResult(**d) for d in docs],
    )


@app.post("/search", response_model=SearchResponse)
def search_post(req: SearchRequest) -> SearchResponse:
    return _search(req.query, req.k)


@app.get("/search", response_model=SearchResponse)
def search_get(q: str, k: int = 5) -> SearchResponse:
    return _search(q, k)
