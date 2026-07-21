"""RAG 생성 파트. retrieval은 아직 mock.

실행: python -m rag.generate           # 자체 점검(API 키 불필요)
      python -m rag.generate --live    # 실제 OpenAI 호출
"""

import os
from dataclasses import dataclass, field

from pydantic import BaseModel

from .prompts import build_messages

MODEL = "gpt-4o-mini"


@dataclass
class Chunk:
    """retrieval 팀과의 인터페이스 제안. 이 4개만 지켜주면 됨.

    id    : 인용에 쓰는 고유 문자열. 권장 형식 "<doc_id>#<chunk_idx>"
    text  : 청크 본문
    score : 유사도. **클수록 유사**. Chroma distance는 (1 - distance)로 변환해서 넘겨줄 것
    meta  : 나머지 전부 (url, title, doc_id, chunk_idx ...). 생성 파트는 읽지 않음
    """

    id: str
    text: str
    score: float = 0.0
    meta: dict = field(default_factory=dict)


class Answer(BaseModel):
    """LLM이 채워서 반환하는 구조 (structured output 스키마)."""

    answer: str
    cited_ids: list[str]
    answerable: bool


@dataclass
class RagResult:
    answer: str
    cited_ids: list[str]
    answerable: bool
    mode: str
    model: str
    chunks: list[Chunk]

    @property
    def cited_chunks(self) -> list[Chunk]:
        """UI에서 근거 문단을 그대로 보여주기 위한 역참조."""
        return [c for c in self.chunks if c.id in self.cited_ids]


def mock_retrieve(question: str, k: int = 3) -> list[Chunk]:
    """retrieval 붙기 전까지 쓰는 더미. MS MARCO 스타일 passage."""
    pool = [
        Chunk(
            "msmarco_8721#0",
            "The Amazon rainforest covers most of the Amazon basin of South "
            "America, spanning 5,500,000 square kilometres across nine nations.",
            0.91,
            {"title": "Amazon rainforest"},
        ),
        Chunk(
            "msmarco_8721#1",
            "Brazil holds about 60 percent of the rainforest, followed by Peru "
            "with 13 percent and Colombia with 10 percent.",
            0.84,
            {"title": "Amazon rainforest"},
        ),
        Chunk(
            "msmarco_1502#3",
            "Deforestation in the Amazon slowed in the early 2010s but the rate "
            "rose again later in the decade.",
            0.62,
            {"title": "Deforestation"},
        ),
    ]
    return pool[:k]


def generate(question: str, chunks: list[Chunk], mode: str = "strict") -> RagResult:
    from openai import OpenAI  # 지연 임포트: 자체 점검은 SDK 없이도 돈다

    completion = OpenAI().chat.completions.parse(
        model=MODEL,
        messages=build_messages(question, chunks, mode),
        response_format=Answer,
        temperature=0,
    )
    parsed = completion.choices[0].message.parsed
    return RagResult(
        answer=parsed.answer,
        cited_ids=[i for i in parsed.cited_ids if any(c.id == i for c in chunks)],
        answerable=parsed.answerable,
        mode=mode,
        model=MODEL,
        chunks=chunks,
    )


def _self_check():
    chunks = mock_retrieve("q")
    msgs = build_messages("Where is the Amazon?", chunks, "strict")
    assert msgs[0]["content"] != build_messages("q", chunks, "lenient")[0]["content"]
    assert "[msmarco_8721#0]" in msgs[1]["content"], "청크 id가 컨텍스트에 없음"
    assert "Question: Where is the Amazon?" in msgs[1]["content"]

    try:
        build_messages("q", chunks, "cot")
    except ValueError:
        pass
    else:
        raise AssertionError("모르는 mode는 거부해야 함")

    # 환각 인용(존재하지 않는 id)은 걸러지고, cited_chunks는 역참조된다
    r = RagResult("a", ["msmarco_8721#1", "없는id"], True, "strict", MODEL, chunks)
    assert [c.id for c in r.cited_chunks] == ["msmarco_8721#1"]
    print("self-check ok")


if __name__ == "__main__":
    import sys

    if "--live" in sys.argv:
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY 없음 (.env 확인)")
        q = "How much of the Amazon rainforest is in Brazil?"
        for mode in ("strict", "lenient"):
            r = generate(q, mock_retrieve(q), mode)
            print(f"\n--- {mode} (answerable={r.answerable}) ---\n{r.answer}")
            print("근거:", r.cited_ids)
    else:
        _self_check()
