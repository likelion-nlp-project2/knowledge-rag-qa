"""V1 새니티 체크 — 4컬럼 지표가 양극단에 제대로 반응하는지 확인.

전수 213건(유료 판정)을 돌리기 전에 반드시 통과시켜야 하는 관문이다.
지표 배선이 틀렸는데 전수를 돌리면, 그럴듯하지만 의미 없는 숫자가 나오고
그게 발표 자료에 박힌다. 여기서 걸러낸다.

확인하는 것 — 정답을 아는 입력을 넣고 기대 방향으로 움직이는지만 본다:

  ContextRecall     검색문서=정답문서   -> 높아야  |  검색문서=무관문서  -> 낮아야
  AnswerCorrectness 답변=reference 그대로 -> 높아야  |  답변=엉뚱한 문장  -> 낮아야

숫자의 절대값이 아니라 '갈리는지'가 판정 기준이다(LLM 판정이라 1.0/0.0 딱 떨어지진 않는다).

  python evaluation/check_metric_sanity.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

PILOT = Path(__file__).resolve().parent.parent / "result" / "reference_answers_pilot.jsonl"

# run_generation_eval.py 와 같은 판정기를 써야 여기 통과가 저기 보증이 된다.
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "gpt-5.4-mini")

# 어떤 질문과도 무관한 더미 — '낮게 나와야 하는' 쪽 입력
IRRELEVANT_CONTEXT = (
    "토마토는 가지과에 속하는 식물이다. 주로 붉은색 열매를 먹으며 "
    "샐러드나 소스로 조리한다. 원산지는 남아메리카 안데스 지역이다."
)
IRRELEVANT_ANSWER = "토마토는 샐러드나 소스로 조리해 먹는 붉은 열매채소입니다."

# 통과 기준: 두 조건의 차이가 이만큼은 벌어져야 지표가 '살아있다'고 본다.
MIN_GAP = 0.3


def load_pilot() -> list[dict]:
    if not PILOT.exists():
        raise SystemExit(
            f"파일럿 라벨이 없습니다: {PILOT}\n"
            "  python evaluation/build_reference_answers.py --limit 5 "
            "--out result/reference_answers_pilot.jsonl"
        )
    rows = [json.loads(line) for line in open(PILOT, encoding="utf-8") if line.strip()]
    labeled = [r for r in rows if r["reference"]]     # NO_ANSWER 건은 제외
    if not labeled:
        raise SystemExit("라벨이 전부 NO_ANSWER 입니다 — 라벨 생성부터 다시 확인하세요.")
    return labeled


def gold_passages(qids_cids: list[str]) -> dict[str, str]:
    """새니티용 정답 문서 본문. 코퍼스에서 필요한 것만 뽑는다."""
    from rag.config import DATA
    path = Path(__file__).resolve().parent.parent / "data" / "ko_miracl_reduced_corpus.jsonl"
    need = set(qids_cids)
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row[DATA.c_id] in need:
                out[row[DATA.c_id]] = row[DATA.c_text]
                if len(out) == len(need):
                    break
    return out


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY 가 필요합니다 (.env)")

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerCorrectness, LLMContextRecall

    rows = load_pilot()
    texts = gold_passages([c for r in rows for c in r["gold_cids"]])
    print(f"파일럿 {len(rows)}건 | 판정기 {JUDGE_MODEL}")

    judge = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    metrics = [LLMContextRecall(llm=judge), AnswerCorrectness(llm=judge, embeddings=emb)]

    def score(label: str, samples: list[dict]) -> dict:
        df = ragas_evaluate(EvaluationDataset.from_list(samples), metrics=metrics).to_pandas()
        got = {c: float(df[c].mean()) for c in ("context_recall", "answer_correctness") if c in df}
        print(f"  {label:12s} " + "  ".join(f"{k}={v:.3f}" for k, v in got.items()))
        return got

    # ── GOOD: 정답 문서를 검색해왔고 답변도 reference 그대로 ──
    good = [
        {
            "user_input": r["question"],
            "retrieved_contexts": [texts[c] for c in r["gold_cids"] if c in texts] or [IRRELEVANT_CONTEXT],
            "response": r["reference"],
            "reference": r["reference"],
        }
        for r in rows
    ]
    # ── BAD: 무관한 문서만 검색됐고 답변도 엉뚱함 ──
    bad = [
        {
            "user_input": r["question"],
            "retrieved_contexts": [IRRELEVANT_CONTEXT],
            "response": IRRELEVANT_ANSWER,
            "reference": r["reference"],
        }
        for r in rows
    ]

    print("\n----- 양극단 비교 -----")
    hi = score("정답 조건", good)
    lo = score("오답 조건", bad)

    print("\n----- 판정 -----")
    ok = True
    for k in ("context_recall", "answer_correctness"):
        if k not in hi or k not in lo:
            print(f"  {k}: 계산 안 됨 — 컬럼명/배선 확인 필요")
            ok = False
            continue
        gap = hi[k] - lo[k]
        mark = "PASS" if gap >= MIN_GAP else "FAIL"
        ok &= gap >= MIN_GAP
        print(f"  [{mark}] {k}: {hi[k]:.3f} vs {lo[k]:.3f} (차이 {gap:+.3f}, 기준 {MIN_GAP})")

    if ok:
        print("\n통과 — 전수 실행으로 진행해도 된다.")
    else:
        raise SystemExit(
            "\n실패 — 지표가 양극단을 못 가른다. 전수(유료) 실행 금지.\n"
            "  reference 내용 / 컬럼명(user_input·retrieved_contexts·response·reference) / "
            "판정 모델을 확인할 것."
        )


if __name__ == "__main__":
    main()
