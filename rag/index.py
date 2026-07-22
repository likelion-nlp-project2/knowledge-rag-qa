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


def _chunk_idx(chunk_id: str) -> int:
    """'985055#3' -> 3. '#' 뒤 숫자가 없으면 0."""
    tail = chunk_id.split("#", 1)[1] if "#" in chunk_id else "0"
    return int(tail) if tail.isdigit() else 0


def fetch_full_documents(collection, hits: List[dict]) -> List[dict]:
    """검색된 k개 청크의 doc_id를 취합해 각 문서의 '전체 청크'를 복원해 돌려준다.

    - hits: retrieve() 결과 (corpus_id, score 사용)
    - 같은 문서의 모든 청크를 chunk 번호 순으로 이어붙여 full text 구성
    - 문서 정렬: 매칭된 청크의 최고 score 순 (검색 관련도 유지)
    - doc_id 기준 조회이므로 metadata 에 doc_id 가 있어야 한다(enrich 실행 후 보장됨).

    반환: [{text, score, metadata{doc_id,title,url,doc_type,categories,matched_chunks,n_chunks}}]
    """
    best: Dict[str, float] = {}
    matched: Dict[str, List[str]] = {}
    order: List[str] = []
    for h in hits:
        did = h["corpus_id"].split("#", 1)[0]
        if did not in best:
            best[did] = h["score"]
            matched[did] = []
            order.append(did)
        best[did] = max(best[did], h["score"])
        matched[did].append(h["corpus_id"])

    # 관련도(최고 score) 높은 문서 순으로 정렬
    order.sort(key=lambda d: best[d], reverse=True)

    hit_by_id = {h["corpus_id"]: h for h in hits}

    docs: List[dict] = []
    for did in order:
        res = collection.get(
            where={"doc_id": did}, include=["documents", "metadatas"]
        )
        ids = res.get("ids", [])
        if ids:
            # enrich 후: 문서의 '전체' 청크를 번호순으로 복원
            items = sorted(
                zip(ids, res["documents"], res["metadatas"]),
                key=lambda t: _chunk_idx(t[0]),
            )
            full_text = "\n".join(doc for _, doc, _ in items)
            meta0 = items[0][2] or {}
            n_chunks = len(items)
        else:
            # baseline fallback: doc_id metadata 없음(enrich 전) → 매칭된 청크로만 구성
            ms = sorted(matched[did], key=_chunk_idx)
            full_text = "\n".join(hit_by_id[c]["text"] for c in ms)
            meta0 = {"title": hit_by_id[ms[0]]["title"]}
            n_chunks = len(ms)

        matched_sorted = sorted(matched[did], key=_chunk_idx)
        docs.append(
            {
                "text": full_text,
                "score": best[did],
                "metadata": {
                    "doc_id": did,
                    "title": meta0.get("title", ""),
                    "url": meta0.get("url", f"https://ko.wikipedia.org/?curid={did}"),
                    "doc_type": meta0.get("doc_type", ""),
                    "categories": meta0.get("categories", ""),
                    "matched_chunks": "|".join(matched_sorted),
                    "n_chunks": n_chunks,
                },
            }
        )
    return docs
