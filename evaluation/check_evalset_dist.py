# ============================================
# 평가셋 질문 길이 분포 점검 — 213개가 길이 축에서 치우쳐 있나? (멘토 피드백 #11)
#
# 피드백 #11은 "평가셋 샘플링은 질문 길이 분포 + 카테고리 분포 두 축을 고려할 것"이다.
# 이 스크립트가 '길이' 축, check_evalset_topics.py 가 '카테고리(doc_type)' 축을 담당한다.
#
# 비교 방식: 평가셋(dev 정답 쿼리 전수 213) 의 길이 분포를
#   (a) dev split 전체 질문, (b) 데이터셋 전체 질문 과 대조한다.
#   평균만 보면 꼬리가 가려지므로 중앙값·사분위·최대까지 함께 본다.
#
# 참고: 현재 평가셋은 dev 에서 정답 있는 질문을 "전수" 쓴다(무작위 표본이 아님).
#   그래서 dev 정답 풀 기준으로는 표본 추출 편향이 원천적으로 없다 — 이 점이
#   길이 축에 대한 가장 강한 답이고, 아래 리포트도 그걸 명시적으로 확인해 준다.
#
# 실행:
#   python evaluation/check_evalset_dist.py
# ============================================

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from rag.config import DATA, SEED
from rag.data import load_qrels, load_queries, sample_pos_queries

OUT_CSV = Path(__file__).resolve().parent / "data" / "evalset_lengths.csv"

# 상위 3개 집중도/고른 정도 대신, 길이는 연속값이라 분포 차이로 판정한다.
# 평균 차이가 전체 표준편차의 이 비율을 넘으면 "치우침 있음"으로 본다(효과크기 기준).
BIAS_EFFECT_SIZE = 0.2


def describe(lengths: pd.Series) -> dict:
    return {
        "n": len(lengths),
        "평균": lengths.mean(),
        "표준편차": lengths.std(ddof=0),
        "최소": lengths.min(),
        "25%": lengths.quantile(0.25),
        "중앙값": lengths.median(),
        "75%": lengths.quantile(0.75),
        "최대": lengths.max(),
    }


def print_row(label: str, d: dict) -> None:
    print(f"  {label:<16} n={d['n']:>5}  평균={d['평균']:>6.1f}  중앙값={d['중앙값']:>5.1f}  "
          f"25~75%={d['25%']:>5.1f}~{d['75%']:<5.1f}  최소~최대={d['최소']}~{d['최대']}")


def main() -> None:
    queries = load_queries(DATA)
    dev = load_qrels(DATA, DATA.dev_split)

    # 다른 평가 스크립트와 동일한 평가셋(dev 정답 쿼리 전수, seed 고정 순서)
    n_pos = dev[dev[DATA.qr_score] > 0][DATA.qr_qid].nunique()
    eval_qids = sample_pos_queries(dev, DATA, n=n_pos, seed=SEED)

    dev_qids = dev[DATA.qr_qid].unique().tolist()

    eval_len = pd.Series([len(queries[q]) for q in eval_qids], dtype=float)
    dev_len = pd.Series([len(queries[q]) for q in dev_qids if q in queries], dtype=float)
    all_len = pd.Series([len(t) for t in queries.values()], dtype=float)

    print(f"===== 평가셋 질문 길이 분포 (글자 수, 평가셋 {len(eval_qids)}개) =====")
    print_row("평가셋(213)", describe(eval_len))
    print_row("dev 전체", describe(dev_len))
    print_row("데이터셋 전체", describe(all_len))

    # 판정: 평가셋 평균이 비교군 평균에서 얼마나 떨어져 있나 (전체 표준편차 기준)
    print("\n--- 치우침 판정 (평균 차이 / 비교군 표준편차) ---")
    biased = False
    for label, ref in (("dev 전체", dev_len), ("데이터셋 전체", all_len)):
        diff = eval_len.mean() - ref.mean()
        sd = ref.std(ddof=0)
        effect = abs(diff) / sd if sd > 0 else 0.0
        flag = "치우침" if effect >= BIAS_EFFECT_SIZE else "차이 미미"
        biased = biased or effect >= BIAS_EFFECT_SIZE
        print(f"  vs {label:<14} 평균차 {diff:+.1f}자  효과크기 {effect:.3f}  -> {flag}")

    print("\n판정:", "길이 축에 치우침 있음 -> 결과 해석 시 명시 필요" if biased
          else "길이 축에 유의미한 치우침 없음")

    # 전수 사용이면 표본 추출 편향은 원천적으로 없다 — 근거로 함께 남긴다.
    if len(eval_qids) == n_pos:
        print(f"근거: 평가셋이 dev 정답 쿼리 {n_pos}개 '전수'라, dev 정답 풀 기준 "
              f"표본 추출 편향은 정의상 0 (부분 표본이 아님).")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "qid": eval_qids,
        "question": [queries[q] for q in eval_qids],
        "length": [len(queries[q]) for q in eval_qids],
    }).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n저장:", OUT_CSV)


if __name__ == "__main__":
    main()
