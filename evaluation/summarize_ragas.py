"""RAGAS 건별 CSV → 프롬프트 A/B 요약표 (실험 마지막 단계).

result/ragas_*.csv 전부 모아 prompt_style 별 평균을 내고
result/ragas_summary.csv 로 저장한다. 보고서 표는 이 파일을 그대로 쓴다.

  python evaluation/summarize_ragas.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULT_DIR = Path(__file__).resolve().parent.parent / "result"


def main() -> None:
    files = sorted(RESULT_DIR.glob("ragas_generations_*.csv"))
    if not files:
        raise SystemExit("result/ 에 ragas_generations_*.csv 가 없습니다. run_ragas_eval.py 먼저 실행하세요.")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    metrics = [c for c in df.select_dtypes("number").columns if c != "qid"]
    summary = (
        df.groupby("prompt_style")
        .agg(n=("qid", "count"), **{c: (c, "mean") for c in metrics})
        .round(4)
    )
    out = RESULT_DIR / "ragas_summary.csv"
    summary.to_csv(out, encoding="utf-8-sig")
    print("입력:", *[f.name for f in files])
    print("저장:", out)
    print(summary)


if __name__ == "__main__":
    main()
