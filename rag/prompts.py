"""생성 파트 프롬프트 단일 소스.

두 층으로 구분한다:
  1) 시스템 프롬프트 — 고정 필수 규칙. 모든 생성 호출에 공통 적용되며 실험 중 바꾸지 않는다.
  2) 생성 프롬프트 — 검색 문서(contexts)를 주입해 실제 user 메시지를 조립하는 층.
     변형(지시형/few-shot)은 PROMPT_BUILDERS 레지스트리로 관리한다.

소비처: rag.generation(rag_answer), rag_engine(Streamlit 앱·배치 생성).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# 1) 시스템 프롬프트 — 고정 필수 규칙
#    ① 참고 문서만을 기반으로 답변  ② 문서에 없으면 해당 항목 위치에서 기권
#    ③ 복합 질문은 원래 순서대로 항목별 독립 처리 (서두 전체 거절 금지)
# ─────────────────────────────────────────────────────────────
SYSTEM_GEN = (
    "당신은 한국어로 답하는 QA assistant입니다. "
    "반드시 주어진 참고 문서만을 기반으로 답변하세요. "
    "문서에서 찾을 수 없는 정보는 '제공된 문서에서 찾을 수 없습니다'라고 답하세요. "
    "답변에 사용한 문서 번호를 [문서 n] 형태로 인용하세요.\n"
    "\n"
    "[복합 질문 처리 및 답변 순서 엄수 규칙]\n"
    "1. 질문 순서 보존: 사용자가 한 번에 2개 이상의 질문을 하면, 반드시 사용자가 "
    "질문한 원래 순서 그대로 항목별로 답변을 작성하세요. 절대로 질문의 순서를 "
    "바꾸거나 섞어서 답변하지 마세요.\n"
    "2. 서두 전체 거절/부재 선언 금지: 일부 질문의 정보가 문서에 없다고 해서, 답변 "
    "시작 부분에 '정보가 없습니다' 같은 전체 총평을 먼저 출력하지 마세요. 정보의 "
    "유무 판단과 대체 정보 안내는 반드시 해당 질문 항목의 순서 위치에서만 수행하세요.\n"
    "3. 항목별 독립 처리: 문서에 정보가 있는 질문은 그 위치에서 정확히 답하세요. "
    "정보가 없는 질문은 그 위치에서 해당 내용은 제공된 문서에서 찾을 수 없다고 "
    "명시한 뒤, 문서에 있는 관련 대체 정보(예: 역대·초대 기록)가 있으면 같은 "
    "위치에서 [문서 n] 인용과 함께 안내하세요."
)

# 질의 재작성용 (생성 규칙과 별개 — 검색 전 단계)
SYSTEM_REWRITE = (
    "당신은 검색 질의 재작성 전문가입니다. 사용자의 모호하거나 추상적인 질문을 "
    "검색엔진에 적합하도록 핵심 개체·키워드 중심의 명확한 한국어 검색 질의로 바꾸세요. "
    "설명 없이 재작성된 질의 한 줄만 출력하세요."
)

# HyDE 재작성용 — 질문 대신 '문서처럼 생긴 가상 답변'을 만들어 dense 검색 질의로 쓴다
SYSTEM_HYDE = (
    "당신은 위키백과 스타일의 백과사전 문서를 작성하는 어시스턴트입니다. "
    "사용자의 질문에 대한 답을 이미 안다고 가정하고, 그 답이 담긴 위키백과 문단을 "
    "2~4문장으로 작성하세요. 실제로 맞는 답인지 모르더라도 사실처럼 서술하고, "
    "'모르겠다'거나 질문을 되묻지 마세요. 다른 설명 없이 문단만 출력하세요."
)

# 하위 호환 별칭 (mode 기반 소비처: run_generation_eval, 노트북)
# "fewshot"도 시스템 규칙은 동일 — 차이는 user 메시지의 few-shot 예시 유무뿐
SYSTEM_PROMPTS = {"strict": SYSTEM_GEN, "basic": SYSTEM_GEN, "fewshot": SYSTEM_GEN}


# ─────────────────────────────────────────────────────────────
# 2) 생성 프롬프트 — 검색 문서를 주입한 user 메시지 조립
#    contexts: [{"corpus_id", "title", "text"}] (rag.index.retrieve 형식)
# ─────────────────────────────────────────────────────────────
def _format_ctx(contexts: List[Dict], snippet_len: Optional[int] = None) -> str:
    return "\n\n".join(
        f"[문서 {i + 1}] (id={c['corpus_id']}, title={c['title']})\n"
        + (c["text"][:snippet_len] if snippet_len else c["text"])
        for i, c in enumerate(contexts)
    )


def build_prompt(question: str, contexts: List[Dict], mode: str = "strict",
                 snippet_len: Optional[int] = None) -> Tuple[str, str]:
    """A) 지시형: 시스템 규칙 + 문서 + 질문."""
    if mode not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown mode: {mode!r}, expected one of {list(SYSTEM_PROMPTS)}")
    user = f"# 참고 문서\n{_format_ctx(contexts, snippet_len)}\n\n# 질문\n{question}\n\n# 답변"
    return SYSTEM_PROMPTS[mode], user


_FEWSHOT = """다음은 답변 형식 예시입니다.

# 예시 1 (근거가 된 문서를 모두 인용, 근거 아닌 문서는 인용하지 않음)
# 참고 문서
[문서 1] (id=ex-a, title=에베레스트산)
에베레스트산은 해발 8,848m로 지구에서 가장 높은 산이다.
[문서 2] (id=ex-b, title=백두산)
백두산은 한반도에서 가장 높은 산이다.
[문서 3] (id=ex-c, title=에베레스트산 등반사)
에베레스트산은 네팔과 중국 티베트 자치구의 국경에 걸쳐 있다.
# 질문
세계에서 가장 높은 산은 어디에 있고 높이는 얼마인가요?
# 답변
세계에서 가장 높은 산은 에베레스트산으로, 높이는 해발 8,848m입니다 [문서 1]. 네팔과 중국 티베트 자치구의 국경에 걸쳐 있습니다 [문서 3].

# 예시 2 (문서에 근거 없음 → 지어내지 말고 거절)
# 참고 문서
[문서 1] (id=ex-d, title=커피)
커피는 커피나무 열매의 씨앗을 볶아 만든 음료다.
# 질문
녹차에 들어있는 카페인 함량은?
# 답변
제공된 문서에서 찾을 수 없습니다.

이제 아래 실제 질문에 위 형식으로 답하세요.
"""


def build_prompt_fewshot(question: str, contexts: List[Dict], mode: str = "strict",
                         snippet_len: Optional[int] = None) -> Tuple[str, str]:
    """B) few-shot: 다중 인용 답변 1개 + 근거없음→거절 1개를 예시로 주입."""
    system, user = build_prompt(question, contexts, mode, snippet_len)
    return system, _FEWSHOT + "\n" + user


# 변형 레지스트리 — 새 변형은 여기 등록
PROMPT_BUILDERS = {"basic": build_prompt, "fewshot": build_prompt_fewshot}
