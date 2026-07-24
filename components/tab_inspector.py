"""탭 2: 검색 과정 살펴보기 — 어떤 문서가 왜 선택/제외됐는지 보여준다."""
from typing import Optional

import pandas as pd
import streamlit as st

from schema import PipelineConfig, RAGResponse

_PASS_STYLE = "background-color: rgba(16, 185, 129, 0.15)"
_FAIL_STYLE = "background-color: rgba(239, 68, 68, 0.12)"


def render(cfg: PipelineConfig) -> None:
    resp: Optional[RAGResponse] = st.session_state.get("last_response")
    if resp is None:
        st.info("💬 질의응답 탭에서 먼저 질문을 하면 검색 과정이 여기에 표시됩니다.")
        return

    # ── 질문 분석 ──
    st.subheader("질문 분석")
    c1, c2 = st.columns(2)
    c1.text_input("입력한 질문", resp.query, disabled=True)
    c2.text_input("검색에 사용된 질문", resp.rewritten_query or "(변환 없음)", disabled=True)

    # ── 관련도 점수 · 단계별 소요 시간 ──
    scores = [c.final_score for c in resp.retrieved_chunks]
    if not scores:
        st.warning("검색된 문서가 없습니다.")
        return
    st.subheader("관련도 점수")
    st.caption("관련도: 문서가 질문과 얼마나 관련 있는지 0~1로 나타낸 점수입니다. 1에 가까울수록 관련이 높습니다.")
    m1, m2, m3 = st.columns(3)
    m1.metric("최고 점수", f"{max(scores):.3f}", help="검색된 문서 중 가장 관련 높은 문서의 점수")
    m2.metric("최저 점수", f"{min(scores):.3f}", help="검색된 문서 중 가장 관련 낮은 문서의 점수")
    m3.metric("평균 점수", f"{sum(scores) / len(scores):.3f}",
              help="평균이 낮으면 질문과 맞는 문서가 적다는 신호입니다")

    t = resp.execution_time_sec
    l1, l2, l3 = st.columns(3)
    l1.metric("질문 변환 (초)", t.get("rewrite", 0.0),
              help="질문을 검색에 유리한 형태로 바꾸는 데 걸린 시간")
    l2.metric("문서 검색 (초)", t.get("retrieval", 0.0),
              help="문서를 찾고 정밀 재정렬까지 마치는 데 걸린 시간")
    l3.metric("답변 생성 (초)", t.get("llm", 0.0),
              help="찾은 문서를 근거로 답변을 작성하는 데 걸린 시간")

    # ── 검색된 문서 목록 ──
    st.subheader("검색된 문서 목록")
    passed_ids = {c.doc_id for c in resp.filtered_chunks}
    df = pd.DataFrame([
        {
            "순위": c.rank,
            "문서 제목": c.metadata.title,
            "관련도": c.final_score,
            "의미 유사도": c.dense_score,
            "상태": "답변에 사용" if c.doc_id in passed_ids else "제외됨",
            "매칭 청크": c.metadata.matched_chunks,
            "근거 청크(±이웃)": c.metadata.window_chunks,
        }
        for c in resp.retrieved_chunks
    ])

    def _highlight(row: pd.Series) -> list:
        style = _PASS_STYLE if row["상태"] == "답변에 사용" else _FAIL_STYLE
        return [style] * len(row)

    st.dataframe(df.style.apply(_highlight, axis=1), use_container_width=True, hide_index=True)
    st.caption(
        f"사용 기준: 관련도 {cfg.min_score:.2f}점 이상인 문서 중 상위 {cfg.top_k_context}개까지 답변에 사용합니다. "
        "'의미 유사도'는 재정렬 전 1차 검색 단계의 점수(참고용)입니다.")
    st.caption(
        "청크 표기: `1234#5` = 문서 1234의 5번 청크. "
        "'매칭 청크'는 검색·재정렬에 걸린 청크, '근거 청크(±이웃)'는 실제 답변 근거로 실린 청크(매칭+앞뒤 이웃)입니다.")
