"""검색 파이프라인: dense 검색 → 메타 필터 → 리랭크 → 윈도우 컨텍스트 조립.

  dense top-N 청크
    → quality_ok / doc_type 필터 (노이즈·메타페이지 제거)
    → 리랭커 재점수 → top-K 청크
    → 각 청크에 ±window 이웃 청크만 붙여 문서별로 병합(문서당·전체 예산 상한)
    → 근거구절(전체문서 아님) 반환

전체문서 복원(rag.index.fetch_full_documents)과 달리, 여기서는 매칭 청크 주변만
잘라 LLM 컨텍스트 폭증을 막는다. 리랭커는 rag.rerank.Reranker 로 갈아끼운다.
"""

from __future__ import annotations

from typing import List, Optional

from .embedding import embed
from .rerank import Reranker

# RAG 근거로 부적합한 doc_type — 리랭크 전에 후보에서 제거한다.
EXCLUDE_DOC_TYPES = {"동음이의", "목록"}


def _idx(chunk_id: str) -> int:
    """'985055#3' -> 3. '#' 뒤 숫자가 없으면 0."""
    tail = chunk_id.split("#", 1)[1] if "#" in chunk_id else "0"
    return int(tail) if tail.isdigit() else 0


def dense_candidates(collection, model, query: str, query_prefix: str,
                     n: int) -> List[dict]:
    """dense 검색으로 상위 n개 청크 후보를 뽑는다(메타데이터 포함)."""
    q_emb = embed(model, [query], query_prefix, batch_size=1)
    res = collection.query(query_embeddings=q_emb.tolist(), n_results=n)
    out: List[dict] = []
    for cid, dist, doc, meta in zip(
        res["ids"][0], res["distances"][0],
        res["documents"][0], res["metadatas"][0],
    ):
        m = meta or {}
        out.append({
            "corpus_id": cid,
            "doc_id": m.get("doc_id", cid.split("#", 1)[0]),
            "dense_score": 1 - dist,
            "text": doc,
            "meta": m,
        })
    return out


def filter_candidates(cands: List[dict],
                      exclude_types=EXCLUDE_DOC_TYPES,
                      require_quality: bool = True) -> List[dict]:
    """노이즈 청크(quality_ok=False)·부적합 doc_type 를 걸러낸다."""
    out = []
    for c in cands:
        m = c["meta"]
        if require_quality and m.get("quality_ok") is False:
            continue
        if m.get("doc_type") in exclude_types:
            continue
        out.append(c)
    return out


def _cap_around(sel_idx: set, wanted: List[int], cap: int) -> List[int]:
    """wanted 를 cap 개로 줄이되, 매칭(sel_idx) 청크 우선 + 가까운 이웃 우선."""
    ranked = sorted(
        wanted,
        key=lambda i: (0 if i in sel_idx else 1,
                       min(abs(i - s) for s in sel_idx)),
    )
    return sorted(ranked[:cap])


def assemble_windows(collection, selected: List[dict], window: int,
                     per_doc_max: int, total_max: int) -> List[dict]:
    """top-K 청크를 문서별로 묶고, 각 매칭 청크에 ±window 이웃을 붙여 근거구절 구성.

    per_doc_max: 한 문서에서 실을 최대 청크 수. total_max: 전체 청크 예산.
    """
    by_doc: dict = {}
    order: List[str] = []
    for c in selected:
        did = c["doc_id"]
        if did not in by_doc:
            by_doc[did] = {
                "score": c["rerank_score"], "dense": c["dense_score"],
                "matched": [], "sel_idx": set(), "meta": c["meta"],
            }
            order.append(did)
        d = by_doc[did]
        d["score"] = max(d["score"], c["rerank_score"])
        d["dense"] = max(d["dense"], c["dense_score"])
        d["matched"].append(c["corpus_id"])
        d["sel_idx"].add(_idx(c["corpus_id"]))

    order.sort(key=lambda did: by_doc[did]["score"], reverse=True)

    results: List[dict] = []
    budget = total_max
    for did in order:
        if budget <= 0:
            break
        d = by_doc[did]
        # 매칭 청크마다 ±window 이웃 인덱스 수집
        idxs = set()
        for si in d["sel_idx"]:
            idxs.update(j for j in range(si - window, si + window + 1) if j >= 0)
        wanted = sorted(idxs)
        # 문서당 상한 + 전체 예산 적용(매칭 청크·근접 우선)
        cap = min(per_doc_max, budget)
        if len(wanted) > cap:
            wanted = _cap_around(d["sel_idx"], wanted, cap)

        ids = [f"{did}#{j}" for j in wanted]
        got = collection.get(ids=ids, include=["documents"])
        idx_text = {_idx(gid): gdoc
                    for gid, gdoc in zip(got.get("ids", []),
                                         got.get("documents", []))}
        used = [i for i in wanted if i in idx_text]
        if not used:
            continue
        text = "\n".join(idx_text[i] for i in used)
        budget -= len(used)

        m = d["meta"]
        results.append({
            "text": text,
            "rerank_score": d["score"],
            "dense_score": d["dense"],
            "metadata": {
                "doc_id": did,
                "title": m.get("title", ""),
                "url": m.get("url", f"https://ko.wikipedia.org/?curid={did}"),
                "doc_type": m.get("doc_type", ""),
                "categories": m.get("categories", ""),
                "matched_chunks": "|".join(
                    sorted(set(d["matched"]), key=_idx)),
                "window_chunks": "|".join(f"{did}#{i}" for i in used),
            },
        })
    return results


def search(collection, model, reranker: Reranker, query: str,
           query_prefix: str = "", *, n: int, top_k: int, window: int,
           per_doc_max: int, total_max: int,
           exclude_types=EXCLUDE_DOC_TYPES) -> List[dict]:
    """전체 파이프라인: dense N → 필터 → 리랭크 → top-K → 윈도우 조립."""
    cands = dense_candidates(collection, model, query, query_prefix, n)
    cands = filter_candidates(cands, exclude_types=exclude_types)
    if not cands:
        return []
    ranked = reranker.rerank(query, cands)
    top = ranked[:top_k]
    return assemble_windows(collection, top, window, per_doc_max, total_max)
