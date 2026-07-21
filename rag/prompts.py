"""프롬프트 템플릿 2종 + 컨텍스트 포맷팅.

비교축: 관대형(LENIENT) vs 엄격형(STRICT).
둘 다 "질문과 같은 언어로 답한다" 규칙을 공유한다.
"""

_COMMON = """You answer questions using the retrieved passages below.
Always answer in the same language as the user's question.
Cite the passage ids you actually used in `cited_ids`."""

LENIENT = f"""{_COMMON}

If the passages don't fully cover the question, you may fall back on your own
general knowledge to give a useful answer. When you do, set `answerable` to
false and say in the answer which part came from outside the passages."""

STRICT = f"""{_COMMON}

Use ONLY the passages. Never use outside knowledge, never guess, never fill
gaps. If the passages do not contain the answer, set `answerable` to false and
make the answer exactly: the information is not in the retrieved passages
(in the question's language). Every claim must trace to a cited passage."""

TEMPLATES = {"lenient": LENIENT, "strict": STRICT}


def format_context(chunks) -> str:
    """청크 리스트를 [id] 본문 형태로 직렬화."""
    return "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)


def build_messages(question: str, chunks, mode: str = "strict") -> list[dict]:
    if mode not in TEMPLATES:
        raise ValueError(f"unknown mode: {mode!r}, expected one of {list(TEMPLATES)}")
    return [
        {"role": "system", "content": TEMPLATES[mode]},
        {
            "role": "user",
            "content": f"Passages:\n{format_context(chunks)}\n\nQuestion: {question}",
        },
    ]
