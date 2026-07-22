"""리랭커 추상화 + 구현 (검색 후 재점수).

여러 방식을 나란히 비교할 수 있도록 공통 인터페이스(Reranker)로 두고 key 로 로드한다.
검색 파이프라인(rag/search.py)은 dense 후보 청크를 이 리랭커로 재정렬한다.

  none               : 리랭커 없음(dense 순서 유지) — 베이스라인
  bge-reranker-v2-m3 : cross-encoder (기본, bge-m3 계열 다국어)
  bge-reranker-base, ko-reranker : 비교 후보

후보(candidate)는 최소 {"text": str, "dense_score": float} 를 갖는 dict 이며,
rerank() 는 각 후보에 "rerank_score" 를 채우고 점수 내림차순으로 정렬해 돌려준다.
"""

from __future__ import annotations

from typing import List

from .config import get_rerank_model


class Reranker:
    """공통 인터페이스. key 로 어떤 방식인지 식별한다."""

    key: str = "none"

    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        raise NotImplementedError


class NoReranker(Reranker):
    """리랭커 없음 — dense 순서를 그대로 유지(rerank_score=dense_score). 베이스라인."""

    key = "none"

    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        for c in candidates:
            c["rerank_score"] = float(c.get("dense_score", 0.0))
        return candidates


class CrossEncoderReranker(Reranker):
    """cross-encoder 리랭커 (query·청크를 함께 인코딩해 관련도 점수 산출)."""

    def __init__(self, key: str, hf_id: str, device: str,
                 max_seq_len: int, fp16: bool):
        from sentence_transformers import CrossEncoder

        self.key = key
        self.model = CrossEncoder(hf_id, max_length=max_seq_len, device=device)
        if fp16 and device == "cuda":
            try:
                self.model.model.half()   # 추론용 fp16 (VRAM/속도)
            except Exception:             # noqa: BLE001 — 실패 시 fp32 유지
                pass

    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        if not candidates:
            return candidates
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(
            pairs, batch_size=32, show_progress_bar=False
        )
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates


def load_reranker(key: str, device: str, fp16: bool = True) -> Reranker:
    """key 에 맞는 리랭커를 로드한다. 'none' 은 모델 로드 없이 즉시 반환."""
    cfg = get_rerank_model(key)
    if cfg.kind == "none":
        return NoReranker()
    return CrossEncoderReranker(cfg.key, cfg.hf_id, device, cfg.max_seq_len, fp16)
