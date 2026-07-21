"""RAG 생성 파트. retrieval은 아직 mock.

실행: python -m rag.generate                 # 자체 점검(API 키 불필요)
      python -m rag.generate --live          # 실제 호출, gpt-4o-mini
      python -m rag.generate --live --pro    # 실제 호출, gpt-4o
"""

import os
from dataclasses import dataclass, field

from pydantic import BaseModel

from .prompts import build_messages

# 모델명은 .env 에서. 없으면 아래 기본값.
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MODEL_PRO = os.getenv("OPENAI_MODEL_PRO", "gpt-4o")  # generate(..., model=MODEL_PRO)


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


def generate(
    question: str, chunks: list[Chunk], mode: str = "strict", model: str = MODEL
) -> RagResult:
    from openai import OpenAI  # 지연 임포트: 자체 점검은 SDK 없이도 돈다

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 가 비어 있음 — .env 를 채울 것")

    completion = OpenAI().chat.completions.parse(
        model=model,
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
        model=model,
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

    # 키가 비어 있으면 401 대신 바로 알아듣게 터져야 함
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        generate("q", chunks)
    except RuntimeError:
        pass
    else:
        raise AssertionError("빈 키면 막아야 함")
    finally:
        if saved:
            os.environ["OPENAI_API_KEY"] = saved

    print("self-check ok")


if __name__ == "__main__":
    import sys

    if "--live" in sys.argv:
        model = MODEL_PRO if "--pro" in sys.argv else MODEL
        # 두 번째 질문이 두 템플릿을 가르는 지점: mock 청크 어디에도 답이 없다.
        questions = [
            ("문서로 답됨/한국어", "아마존 열대우림의 몇 퍼센트가 브라질에 있나요?"),
            ("문서에 없음", "When was the Amazon rainforest first mapped?"),
        ]
        for label, q in questions:
            print(f"\n===== [{label}] {q}")
            for mode in ("strict", "lenient"):
                r = generate(q, mock_retrieve(q), mode, model)
                print(f"--- {mode} / {r.model} (answerable={r.answerable})")
                print(f"{r.answer}\n근거: {r.cited_ids}")
    else:
        _self_check()
