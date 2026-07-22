"""ChromaDB 색인/검색.

- 인메모리(chromadb.Client): 파인튜닝 비교·노트북용 (build_collection)
- 영속 서버(chromadb.HttpClient): Docker 배포·검색 API용 (connect/get_or_create_collection)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from .embedding import embed


def connect(host: str, port: int):
    """영속 Chroma 서버에 연결한다 (Docker 서비스명/포트)."""
    return chromadb.HttpClient(host=host, port=port)


def get_or_create_collection(client, name: str):
    """코사인 공간 컬렉션을 가져오거나 만든다 (ingest/검색 공용)."""
    return client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def build_collection(
    client,
    name: str,
    model: SentenceTransformer,
    cids: List[str],
    corpus_text: Dict[str, str],
    corpus_title: Optional[Dict[str, str]] = None,
    passage_prefix: str = "",
    batch_size: int = 128,
):
    """cids 순서대로 임베딩해 컬렉션에 적재한다. corpus_title이 있으면 title도 메타데이터로 저장."""
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    texts = [corpus_text[c] for c in cids]
    embs = embed(model, texts, passage_prefix, batch_size=batch_size, show_progress=True)
    for i in range(0, len(cids), batch_size):
        batch_ids = cids[i : i + batch_size]
        metadatas = None
        if corpus_title is not None:
            metadatas = [{"title": corpus_title.get(c, "")} for c in batch_ids]
        collection.add(
            ids=batch_ids,
            embeddings=embs[i : i + batch_size].tolist(),
            documents=texts[i : i + batch_size],
            metadatas=metadatas,
        )
    return collection


def retrieve(
    collection,
    model: SentenceTransformer,
    query_text: str,
    query_prefix: str = "",
    k: int = 10,
) -> List[dict]:
    """score = 1 - distance (클수록 유사). title/전체 text가 색인돼 있으면 함께 반환."""
    q_emb = embed(model, [query_text], query_prefix, batch_size=1)
    res = collection.query(query_embeddings=q_emb.tolist(), n_results=k)

    out = []
    for cid, dist, doc, meta in zip(
        res["ids"][0], res["distances"][0], res["documents"][0], res["metadatas"][0]
    ):
        out.append(
            {
                "corpus_id": cid,
                "score": 1 - dist,
                "title": (meta or {}).get("title", ""),
                "text": doc,
            }
        )
    return out
