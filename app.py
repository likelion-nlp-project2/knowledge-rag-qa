"""지식기반 QA(RAG) — 메인 Streamlit 진입점 (UI & 상태 관리).

실행: streamlit run app.py
"""
import streamlit as st

from components import sidebar, tab_inspector, tab_qa

st.set_page_config(page_title="지식기반 QA(RAG)", page_icon="💬", layout="centered")

# ── 세션 상태 초기화 ──
st.session_state.setdefault("chat_history", [])   # [{"query": .., "response": RAGResponse}] 시간순
st.session_state.setdefault("last_response", None)  # 검색 과정 뷰가 소비하는 최근 RAGResponse

# ── 사이드바 → 현재 검색 옵션 ──
cfg = sidebar.render()

st.title("💬 지식기반 QA(RAG)")
st.caption("위키백과 문서를 근거로 답변하고, 참고한 문서를 함께 보여드립니다.")

# ── 메인: 챗봇 ──
tab_qa.render(cfg)

# ── 보조: 최근 질문의 검색 과정 (접이식) ──
if st.session_state.last_response is not None:
    with st.expander("🔎 검색 과정 살펴보기 (최근 질문 기준)"):
        tab_inspector.render(cfg)
