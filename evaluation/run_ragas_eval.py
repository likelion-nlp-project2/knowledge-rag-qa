"""RAGAS 평가 (실험 3단계) — 샘플링 + GPT-5.4-mini judge.

generate_answers.py 산출물에서 N건(기본 150)만 seed 고정 랜덤 샘플링해
faithfulness(충실도) / answer relevancy 를 평가한다.
건별 점수는 result/ragas_*.csv, 평균은 콘솔에 출력.

- judge는 gpt-5.4-mini로 실험 전체 고정 — 바꾸면 이전 결과와 비교 불가하므로 상수로 둔다.
- 비용 주의: 여기가 제일 비싸다. 150건 ≈ $1 안팎 (GPT-5.4-mini). 200건 초과는 차단.
- 사전 조건: .env 에 OPENAI_API_KEY.

  python evaluation/run_ragas_eval.py --n 3                                # 스모크
  python evaluation/run_ragas_eval.py --input result/generations_fewshot.jsonl --n 150
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "gpt-5.4-mini"             # 평가 전용 고정 judge (일관성)
JUDGE_EMBED = "text-embedding-3-small"   # answer relevancy용 임베딩 (저비용)
MAX_EVAL = 200                           # 예산 가이드: 평가는 100~200건 이내


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGAS 샘플 평가 (GPT-4o judge)")
    ap.add_argument("--input", default="result/generations_fewshot.jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.n > MAX_EVAL:
        sys.exit(f"평가는 최대 {MAX_EVAL}건(예산 가이드). --n 을 줄여주세요.")

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    # 빈 응답(스레숄드 미달로 생성 건너뜀)은 faithfulness 정의가 안 되므로 제외
    rows = [r for r in rows if r["answer"] and r["contexts"]]
    random.Random(args.seed).shuffle(rows)
    sample = rows[: args.n]
    print(f"{args.input}: 유효 {len(rows)}건 중 {len(sample)}건 샘플 "
          f"(seed={args.seed}, judge={JUDGE_MODEL})")

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy

    judge = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=JUDGE_EMBED))
    ds = EvaluationDataset.from_list([
        {"user_input": r["question"], "response": r["answer"],
         "retrieved_contexts": r["contexts"]}
        for r in sample
    ])
    result = evaluate(ds, metrics=[Faithfulness(llm=judge),
                                   ResponseRelevancy(llm=judge, embeddings=emb)])

    df = result.to_pandas()
    df.insert(0, "qid", [r["qid"] for r in sample])
    df.insert(1, "prompt_style", [r.get("prompt_style", "") for r in sample])
    out = Path(__file__).resolve().parent.parent / "result" / (
        f"ragas_{Path(args.input).stem}_n{len(sample)}_seed{args.seed}.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("건별 점수 저장:", out)
    print("\n===== 평균 (NaN 제외) =====")
    print(df.select_dtypes("number").mean().round(4))


if __name__ == "__main__":
    main()
