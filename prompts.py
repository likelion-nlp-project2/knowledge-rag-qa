# RAG 생성 프롬프트 모음.
# 노트북(ko_miracl_rag_generation.ipynb)이 GitHub raw로 이 파일을 받아 사용한다.
# 수정 → dh 브랜치 push → 노트북의 프롬프트 로드 셀 재실행으로 반영.

SYSTEM_GEN = ("당신은 주어진 참고 문서를 근거로 한국어로 답하는 assistant입니다. "
              "답변은 한국어로 작성하세요. 고유명사·인명·회사명·전문용어 등은 필요하면 원어를 쓰거나 병기해도 되지만, "
              "문장 자체가 다른 언어로 넘어가면 안 됩니다. "
              "특히 중국어 문장을 출력하지 마세요. 중국어로 표현하고 싶은 내용이 있으면 반드시 한국어로 바꿔 답하세요. "
              "문서에 없는 내용은 지어내지 말고, 근거가 없으면 '제공된 문서에서 찾을 수 없습니다'라고 답하세요. "
              "답변에 사용한 문서 번호를 [문서 n] 형태로 인용하세요.")


def _format_ctx(contexts, snippet_len=800):
    return "\n\n".join(
        f"[문서 {i+1}] (id={c['corpus_id']}, title={c['title']})\n{c['text'][:snippet_len]}"
        for i, c in enumerate(contexts))


# ── A) 베이스라인: 지시만 ──
def build_prompt(question, contexts, snippet_len=800):
    user = f"# 참고 문서\n{_format_ctx(contexts, snippet_len)}\n\n# 질문\n{question}\n\n# 답변"
    return SYSTEM_GEN, user


# ── B) Few-shot: 다중 인용 답변 1개 + 근거없음→거절 1개를 예시로 주입 ──
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


def build_prompt_fewshot(question, contexts, snippet_len=800):
    user = (_FEWSHOT
            + f"\n# 참고 문서\n{_format_ctx(contexts, snippet_len)}"
            + f"\n\n# 질문\n{question}\n\n# 답변")
    return SYSTEM_GEN, user


# 평가에서 비교할 프롬프트 레지스트리 — 새 변형은 여기 등록
PROMPTS = {
    "A_baseline": build_prompt,
    "B_fewshot": build_prompt_fewshot,
}
