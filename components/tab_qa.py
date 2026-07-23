"""질의응답 챗봇 — 대화는 시간순으로 쌓이고 입력창은 화면 하단에 고정된다."""
import requests
import streamlit as st

from rag_engine import run_pipeline, stream_tokens
from schema import PipelineConfig, RAGResponse


def _render_citations(resp: RAGResponse) -> None:
    n = len(resp.filtered_chunks)
    with st.expander(f"📚 참고한 문서 ({n}개)"):
        for c in resp.filtered_chunks:
            st.markdown(f"**[{c.rank}] {c.metadata.title}** (관련도: {c.final_score:.2f})")
            st.markdown(f"> {c.text}")
            if c.metadata.url:
                st.caption(f"출처: {c.metadata.url}")


def _render_assistant(resp: RAGResponse, cfg: PipelineConfig, stream: bool = False) -> None:
    if not resp.filtered_chunks:
        st.warning(
            "⚠️ 질문과 충분히 관련된 문서를 찾지 못했습니다. "
            "부정확한 답변을 드리지 않기 위해 답변 생성을 건너뛰었습니다.")
        return
    if stream:
        st.write_stream(stream_tokens(resp.generated_answer))
    else:
        st.markdown(resp.generated_answer)
    _render_citations(resp)


def render(cfg: PipelineConfig) -> None:
    """지난 대화를 시간순으로 렌더하고, 새 질문은 맨 아래에 이어 붙인다."""
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown("안녕하세요! 궁금한 것을 물어보시면 문서를 찾아 근거와 함께 답해 드릴게요. 🙂")

    for pair in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(pair["query"])
        with st.chat_message("assistant"):
            _render_assistant(pair["response"], cfg)

    # st.chat_input은 화면 하단에 자동 고정된다
    if query := st.chat_input("무엇이든 물어보세요"):
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            resp = None
            try:
                resp = run_pipeline(query, cfg)
            except requests.RequestException as e:
                st.error(f"백엔드 연결 오류입니다. 검색/LLM API 상태를 확인해 주세요: {e}")
            except Exception as e:  # 내부 오류가 앱을 죽이지 않도록
                st.error(f"오류가 발생했습니다: {e}")
            if resp is not None:
                _render_assistant(resp, cfg, stream=True)
                st.session_state.chat_history.append({"query": query, "response": resp})
                st.session_state.last_response = resp
