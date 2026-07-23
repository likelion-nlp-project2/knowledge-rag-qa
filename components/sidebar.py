"""사이드바: 검색·생성 옵션 컨트롤."""
import streamlit as st

from rag_engine import get_backend_health
from schema import PipelineConfig

_PROMPT_LABELS = {"fewshot": "예시 포함 (권장)", "basic": "기본"}


def render() -> PipelineConfig:
    """사이드바를 그리고 현재 컨트롤 값을 PipelineConfig로 반환한다."""
    with st.sidebar:
        st.title("⚙️ 설정")

        st.subheader("검색 설정")
        top_k_retrieval = st.slider(
            "검색할 문서 수", 5, 50, 20,
            help="질문과 관련해 1차로 찾아올 문서 개수입니다. "
                 "많이 찾을수록 놓치는 문서가 줄지만 검색이 느려질 수 있습니다.")
        top_k_context = st.slider(
            "답변에 사용할 문서 수", 1, 10, 5,
            help="찾은 문서 중 실제로 답변 작성에 참고할 상위 문서 개수입니다. "
                 "많을수록 근거가 풍부해지지만 응답이 느려지고 비용이 늘어납니다.")

        st.subheader("답변 품질")
        min_score = st.slider(
            "최소 관련도 점수", 0.0, 1.0, 0.55, 0.01,
            help="이 점수보다 관련도가 낮은 문서는 답변에 사용하지 않습니다. "
                 "높이면 더 확실한 근거만 쓰는 대신 '찾을 수 없음' 답변이 늘어납니다.")
        use_reranker = st.checkbox(
            "검색 결과 정밀 재정렬", value=True,
            help="찾은 문서를 한 번 더 검토해 더 정확한 순서로 배열합니다. "
                 "끄면 빨라지지만 관련 문서가 뒤로 밀릴 수 있습니다.")
        prompt_style = st.selectbox(
            "답변 프롬프트", options=list(_PROMPT_LABELS),
            format_func=_PROMPT_LABELS.get,
            help="'예시 포함'은 모범 답변 예시를 함께 제시해 인용·거절 형식을 더 잘 지키고, "
                 "'기본'은 지시문만 사용해 가볍고 빠릅니다.")

        with st.expander("ℹ️ 시스템 정보"):
            health = get_backend_health()
            if health:
                st.metric("적재된 문단 수", f"{health.get('chunks', 0):,}")
                st.caption(f"임베딩: {health.get('model')} · 리랭커: {health.get('default_reranker')}")
            else:
                st.warning("검색 API에 연결할 수 없습니다.")

    return PipelineConfig(
        top_k_retrieval=top_k_retrieval,
        top_k_context=top_k_context,
        min_score=min_score,
        use_reranker=use_reranker,
        prompt_style=prompt_style,
    )
