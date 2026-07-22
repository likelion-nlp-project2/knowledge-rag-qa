# ============================================
# 리트리버 파인튜닝 전/후 비교 — seed 3개 평균 ± 표준편차 (계획서 4.3 신뢰성)
#
# rag/cli.py 의 run_compare(cfg, seed)를 seed만 바꿔 3회 호출해 집계한다.
# (파이프라인 코드는 수정하지 않음 — 평가 lane 에서 반복·집계만 담당)
#
# 주의:
# - 파인튜닝 학습을 seed당 1회씩 총 3회 수행하므로 GPU 학습 시간이 3배.
#   실행 전 파인튜닝 담당과 자원 사용 조율할 것.
# - run_compare는 seed로 train/eval 쿼리 샘플링까지 바꾸므로, 여기서의 std는
#   (파인튜닝 학습 변동 + 쿼리 샘플링 변동)을 합친 전체 변동성이다.
#   전/후(before/after)는 각 seed 안에서 동일 평가셋으로 비교됨(계획서의 '동일 평가셋 고정').
#
# 실행:
#   SMOKE=1 python evaluation/run_finetune_compare_eval.py   # 구조 확인용(소량·빠름)
#   python evaluation/run_finetune_compare_eval.py            # 실제 보고용(3회 풀 학습)
# ============================================

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from rag.cli import run_compare
from rag.config import SEED, FinetuneConfig

SMOKE = bool(os.environ.get("SMOKE"))
SEEDS = [SEED] if SMOKE else [SEED, SEED + 1, SEED + 2]

OUT_CSV = Path(__file__).resolve().parent / "data" / "finetune_compare_seeds.csv"


def make_cfg() -> FinetuneConfig:
    if not SMOKE:
        return FinetuneConfig()
    # 스모크: 학습/평가/코퍼스를 소량으로 줄여 구조만 빠르게 확인 (숫자는 무의미)
    return FinetuneConfig(n_train_queries=30, n_eval_queries=10, neg_pool_size=300)


def aggregate(per_seed: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """seed별 비교표(before/after/Δ/개선%)를 지표(행)별 mean ± std 로 집계."""
    stacked = pd.concat(per_seed, names=["seed"])           # (seed, metric) 멀티인덱스
    grouped = stacked.groupby(level=1, sort=False)
    mean, std = grouped.mean(), grouped.std(ddof=0)
    out = pd.DataFrame(index=mean.index)
    for col in ["before", "after", "Δ(after-before)", "개선%"]:
        out[f"{col} mean"] = mean[col]
        out[f"{col} std"] = std[col]
    return out


if __name__ == "__main__":
    cfg = make_cfg()
    print(f"파인튜닝 전/후 비교: seeds={SEEDS} (SMOKE={SMOKE})")
    print(f"설정: train {cfg.n_train_queries} / eval {cfg.n_eval_queries} / neg_pool {cfg.neg_pool_size}")

    per_seed: dict[int, pd.DataFrame] = {}
    for s in SEEDS:
        print(f"\n===== run_compare (seed={s}) — 파인튜닝 학습 포함, 오래 걸림 =====")
        per_seed[s] = run_compare(cfg, seed=s)
        print(per_seed[s].round(4))

    summary = aggregate(per_seed)
    print(f"\n===== 파인튜닝 전/후 (seed {len(SEEDS)}개 평균 ± 표준편차) =====")
    print(summary.round(4))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, encoding="utf-8-sig")
    print("\n집계 저장:", OUT_CSV)
