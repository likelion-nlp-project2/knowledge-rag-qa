"""정답 답변 라벨(reference) 생성 — RAGAS ground_truth 컬럼용.

Ko-miracl에는 정답 '문서'(qrels)만 있고 정답 '답변 문장'이 없다. 그래서
Faithfulness/ResponseRelevancy(3컬럼 지표)까지만 계산되고, Context Recall·
Answer Correctness(4컬럼 지표)는 계산할 수 없다. 이 스크립트가 그 4번째 컬럼을 만든다.

만드는 방식: **정답 문서(gold)만 보여주고** 상위 LLM에게 답변을 쓰게 한다(silver label).

  ⚠️ 검색 결과가 아니라 gold 문서로 만든다는 점이 핵심이다.
     검색이 엉뚱한 문서를 가져와도 reference는 정답 문서 기반이어야
     Context Recall이 "검색이 정답 근거를 못 가져왔다"를 잡아낸다.
     검색 결과로 만들면 검색 실패가 정답에 흡수돼 지표가 눈이 먼다.

  ⚠️ 라벨 모델은 판정 모델(RAGAS judge)과 달라야 한다. 같으면 자기 문체에
     후한 점수를 주는 자기 채점 편향이 생긴다.
       라벨 gpt-5.4  >  판정 gpt-5.4-mini  >  생성 gpt-4o-mini

  ⚠️ reference에는 [문서 n] 인용을 넣지 않는다. Context Recall이 reference를
     문장 단위로 쪼개 검색 문서와 대조하는데, 인용 표기가 섞이면 노이즈가 된다.

중단해도 안전하다 — 이미 만든 qid는 건너뛰고 이어서 붙인다(213건 API 호출 재과금 방지).

  python evaluation/build_reference_answers.py --limit 5    # 파일럿(권장)
  python evaluation/build_reference_answers.py              # 전수 213
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.config import DATA, SEED
from rag.data import build_gold, load_qrels, load_queries, sample_pos_queries
from rag.llm import chat

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "ko_miracl_reduced_corpus.jsonl"
OUT_PATH = Path(__file__).resolve().parent.parent / "result" / "reference_answers.jsonl"

# 판정기(gpt-5.4-mini)보다 상위. 팀이 바꾸면 환경변수로 덮어쓴다.
LABEL_MODEL = os.environ.get("LABEL_MODEL", "gpt-5.4")

# gold 문서에 답이 없을 때 LLM이 내도록 지시하는 표식. 이런 질문은 억지 라벨이
# 지표를 오염시키므로 reference=null 로 기록하고 집계에서 제외한다.
NO_ANSWER = "NO_ANSWER"

_SYSTEM = (
    "당신은 QA 평가용 정답(ground truth) 답변을 작성하는 전문가입니다. "
    "주어진 참고 문서는 이 질문의 '정답 문서'로 이미 검증된 것입니다.\n"
    "\n"
    "규칙:\n"
    "1. 참고 문서에 있는 내용만으로 질문에 답하세요. 문서 밖 지식을 더하지 마세요.\n"
    "2. 질문이 묻는 것만 간결하게 답하세요. 보통 1~3문장이면 충분합니다.\n"
    "3. '[문서 1]' 같은 인용 표기를 넣지 마세요. 답변 문장만 쓰세요.\n"
    "4. '문서에 따르면', '제공된 자료에서는' 같은 서두를 붙이지 마세요. "
    "사실을 그대로 서술하세요.\n"
    f"5. 참고 문서를 다 읽어도 질문에 답할 수 없으면, 다른 말 없이 {NO_ANSWER} 만 출력하세요."
)


def load_gold_texts(gold_cids: set[str]) -> dict[str, tuple[str, str]]:
    """코퍼스 jsonl을 한 번 훑어 정답 문서만 뽑는다. cid -> (title, text)."""
    if not CORPUS_PATH.exists():
        raise SystemExit(f"코퍼스 파일이 없습니다: {CORPUS_PATH}")
    out: dict[str, tuple[str, str]] = {}
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cid = row[DATA.c_id]
            if cid in gold_cids:
                out[cid] = (row.get(DATA.c_title) or "", row[DATA.c_text])
                if len(out) == len(gold_cids):
                    break
    return out


def build_user_prompt(question: str, passages: list[tuple[str, str]]) -> str:
    ctx = "\n\n".join(f"[{title}]\n{text}" for title, text in passages)
    return f"# 참고 문서\n{ctx}\n\n# 질문\n{question}\n\n# 정답 답변"


def load_done(path: Path) -> set[str]:
    """이미 라벨을 만든 qid (실패해서 중단됐을 때 이어서 하려고)."""
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["qid"] for line in f if line.strip()}


def verify(out_path: Path, threshold: float) -> None:
    """V2 근거성 검사 — 만든 라벨이 정답 문서를 벗어나지 않았는지 채점한다.

    라벨을 LLM이 만들었으므로 라벨 자체가 지어낸 내용일 수 있다. 그대로 두면
    '틀린 정답'으로 시스템을 채점하게 된다. gold 문서를 근거로 Faithfulness를
    매겨, 낮은 건은 지우고 재생성하도록 목록을 뽑는다(재생성=이 스크립트 재실행).
    """
    rows = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    labeled = [r for r in rows if r.get("reference")]
    if not labeled:
        raise SystemExit("검사할 라벨이 없습니다.")

    gold_texts = load_gold_texts({c for r in labeled for c in r["gold_cids"]})

    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    judge_model = os.environ.get("RAGAS_JUDGE_MODEL", "gpt-5.4-mini")
    judge = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    print(f"근거성 검사: 라벨 {len(labeled)}건 | 판정기 {judge_model} | 임계값 {threshold}")

    ds = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "response": r["reference"],
            # 검색 결과가 아니라 '정답 문서'를 근거로 채점한다 — 라벨이 이 문서를
            # 벗어났는지가 관심사이기 때문이다.
            "retrieved_contexts": [gold_texts[c][1] for c in r["gold_cids"] if c in gold_texts],
        }
        for r in labeled
    ])
    df = ragas_evaluate(ds, metrics=[Faithfulness(llm=judge)]).to_pandas()
    df.insert(0, "qid", [r["qid"] for r in labeled])

    bad = df[df["faithfulness"] < threshold]
    print(f"\n평균 근거성: {df['faithfulness'].mean():.4f}")
    print(f"임계값 미달: {len(bad)}/{len(df)}건")
    for _, row in bad.iterrows():
        q = next(r for r in labeled if r["qid"] == row["qid"])
        print(f"  [{row['faithfulness']:.2f}] {q['question']}\n         -> {q['reference'][:70]}")

    report = out_path.with_name(out_path.stem + "_verify.csv")
    df.to_csv(report, index=False, encoding="utf-8-sig")
    print(f"\n저장: {report}")
    if len(bad):
        print(f"  * 미달 건은 {out_path.name} 에서 해당 줄을 지우고 이 스크립트를 다시 돌리면 재생성된다.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="앞에서 n개만 생성(파일럿). 미지정이면 전수 213건")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--verify", action="store_true",
                    help="생성 대신, 이미 만든 라벨의 근거성(V2)을 검사한다")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="근거성 임계값 (기본 0.7). 미달 라벨을 목록으로 뽑는다")
    args = ap.parse_args()
    out_path = Path(args.out)

    if args.verify:
        verify(out_path, args.threshold)
        return

    queries = load_queries(DATA)
    dev_qrels = load_qrels(DATA, DATA.dev_split)

    # run_generation_eval.py 와 완전히 같은 평가셋·같은 순서여야 qid로 조인된다.
    qids = sample_pos_queries(dev_qrels, DATA, n=10**9, seed=SEED)
    if args.limit:
        qids = qids[: args.limit]

    gold = build_gold(dev_qrels, DATA, qids)
    gold_cids = {c for rels in gold.values() for c, s in rels.items() if s > 0}
    print(f"평가셋 {len(qids)}개 | 정답 문서 {len(gold_cids)}개 | 라벨 모델 {LABEL_MODEL}")

    gold_texts = load_gold_texts(gold_cids)
    missing = gold_cids - gold_texts.keys()
    if missing:
        print(f"경고: 정답 문서 {len(missing)}개가 코퍼스에 없습니다 — 해당 질문은 근거가 부족해집니다.")

    done = load_done(out_path)
    todo = [q for q in qids if q not in done]
    print(f"이미 완료 {len(done)}건 → 이번에 {len(todo)}건 생성")
    if not todo:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_noanswer = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, qid in enumerate(todo, 1):
            passages = [gold_texts[c] for c in gold[qid] if c in gold_texts and gold[qid][c] > 0]
            if not passages:
                print(f"  [{i}/{len(todo)}] {qid}: 정답 문서 본문 없음 — 건너뜀")
                continue

            answer = chat(_SYSTEM, build_user_prompt(queries[qid], passages),
                          max_tokens=400, temperature=0.0, model=LABEL_MODEL)
            is_no = answer.strip().startswith(NO_ANSWER)
            n_noanswer += is_no
            n_ok += not is_no

            f.write(json.dumps({
                "qid": qid,
                "question": queries[qid],
                # NO_ANSWER 는 null 로 기록 — 재시도는 막되 집계에선 빠진다
                "reference": None if is_no else answer.strip(),
                "gold_cids": [c for c in gold[qid] if gold[qid][c] > 0],
                "n_gold_passages": len(passages),
                "label_model": LABEL_MODEL,
            }, ensure_ascii=False) + "\n")
            f.flush()   # 중단돼도 여기까지는 남는다

            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] 라벨 {n_ok} / 답없음 {n_noanswer}")

    print(f"\n저장: {out_path}")
    print(f"라벨 {n_ok}건, 답 없음(제외) {n_noanswer}건")
    if n_noanswer:
        print("  * '답 없음'은 정답 문서에 답이 안 담긴 경우다. 비율이 높으면 "
              "qrels 품질이나 청킹을 의심할 것.")


def _self_check() -> None:
    """API 없이 도는 부분만 검증."""
    p = build_user_prompt("영조는 몇 대 왕인가?", [("조선 영조", "영조는 제21대 왕이다.")])
    assert "# 질문" in p and "영조는 제21대 왕이다." in p and "[조선 영조]" in p
    assert NO_ANSWER in _SYSTEM, "답 없음 표식이 시스템 프롬프트에 있어야 한다"
    assert "인용" in _SYSTEM, "reference에 [문서 n]이 섞이면 Context Recall이 오염된다"

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "r.jsonl"
        assert load_done(f) == set(), "파일 없으면 빈 집합"
        f.write_text('{"qid":"a","reference":"x"}\n{"qid":"b","reference":null}\n', encoding="utf-8")
        assert load_done(f) == {"a", "b"}, "null 라벨도 완료로 쳐야 재시도를 안 한다"
    print("build_reference_answers self-check ok")


if __name__ == "__main__":
    if os.environ.get("SELFCHECK"):
        _self_check()
    else:
        main()
