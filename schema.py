"""Pydantic models for Query, Metadata, and Context Chunks.

검색 API(rag/server.py) → RAG 생성 → Streamlit 파이프라인이 공유하는 타입 계약.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """검색 API가 돌려주는 문서 단위 메타데이터 (rag/server.py ResultMeta와 동일)."""

    doc_id: str
    title: str
    url: Optional[str] = None
    doc_type: str = ""
    categories: str = ""
    matched_chunks: str = ""   # 리랭크 상위로 매칭된 청크 id, '|' join
    window_chunks: str = ""    # 실제 근거구절에 실린 청크(매칭+이웃), '|' join


class RetrievedChunk(BaseModel):
    """검색 파이프라인(dense→필터→리랭크→윈도우)을 거친 문서 단위 근거구절."""

    doc_id: str
    text: str
    final_score: float                 # 정규화 점수 (스레숄드/표시용, 클수록 유사)
    rerank_score: float                # 리랭커 원점수 (정렬 기준)
    dense_score: Optional[float] = None
    rank: int
    metadata: ChunkMetadata


class RAGResponse(BaseModel):
    """파이프라인 1회 실행의 전체 결과 (디버깅 뷰가 소비)."""

    query: str
    rewritten_query: Optional[str] = None
    generated_answer: str
    retrieved_chunks: List[RetrievedChunk]   # 스레숄드 적용 전 전체
    filtered_chunks: List[RetrievedChunk]    # LLM 컨텍스트로 전달된 것
    execution_time_sec: Dict[str, float]     # {"rewrite": .., "retrieval": .., "llm": ..}


class PipelineConfig(BaseModel):
    """사이드바 컨트롤 → 파이프라인으로 전달되는 하이퍼파라미터."""

    top_k_retrieval: int = 20   # 검색 API에 요청할 문서 수 (k)
    top_k_context: int = 5      # 스레숄드 통과 후 LLM에 넘길 문서 수
    min_score: float = 0.55     # final_score 기준 최소 관련도
    use_reranker: bool = True   # False면 rerank="none" (dense 순서 유지)
    prompt_style: str = "fewshot"   # "basic" | "fewshot"
