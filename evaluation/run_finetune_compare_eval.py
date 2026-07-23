# ============================================
# 리트리버 파인튜닝 전/후 비교 — seed 3개 평균 ± 표준편차 (계획서 4.3 신뢰성)
#
# rag/cli.py 의 run_compare(cfg, seed)를 seed만 바꿔 3회 호출해 집계한다.
# (파이프라인 코드는 수정하지 않음 — 평가 lane 에서 반복·집계만 담당)
#
# 주의:
# - 파인튜닝 학습을 seed당 1회씩 총 3회 수행하므로 GPU 학습 시간이 3배.
#   실행 전 파인튜닝 담당과 자원 사용 조율할 것.
# - 코퍼스를 검색 평가와 같은 20만 규모로 맞췄다(NEG_POOL_FULL). HF 스트리밍이라 수집이
#   오래 걸리므로, 구조만 볼 때는 SMOKE=1 또는 NEG_POOL=5000 처럼 줄여서 돌릴 것.
# - 평가셋은 dev 정답 쿼리 "전수"(213개)로 고정한다. sample_pos_queries가 qids[:n]이라
#   n을 전수로 주면 seed가 바뀌어도 뽑히는 집합이 같아진다(순서만 달라지고 지표는 평균이라 무관).
#   덕분에 여기서의 std는 '평가셋이 흔들려서 생긴 변동'이 아니라 순수 학습 변동만 반영하고,
#   계획서 4.3의 '파인튜닝 전/후 동일 평가셋 고정'도 seed 간에까지 성립한다.
#   (학습 쿼리는 seed마다 달라지며, 이는 의도된 학습 변동이다.)
#
# 실행:
#   SMOKE=1 python evaluation/run_finetune_compare_eval.py   # 구조 확인용(소량·빠름)
#   python evaluation/run_finetune_compare_eval.py           # 보고용(3회 풀 학습 — 오래 걸림)
# ============================================

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

from rag.cli import run_compare
from rag.config import DATA, SEED, FinetuneConfig
from rag.data import load_qrels

SMOKE = bool(os.environ.get("SMOKE"))
SEEDS = [SEED] if SMOKE else [SEED, SEED + 1, SEED + 2]

OUT_CSV = Path(__file__).resolve().parent / "data" / "finetune_compare_seeds.csv"


# 검색 평가(run_retrieval_eval.py)가 색인하는 코퍼스 크기에 맞춘다.
#
# 왜 중요한가: 검색 지표는 "정답을 상위 k개에 올렸나"인데, 난이도는 인덱스에 방해 문서가
# 몇 개 있느냐로 정해진다. 기본값 neg_pool_size=3000 을 그대로 쓰면 파인튜닝 비교는
# 약 4천 개 코퍼스에서, 검색 평가는 20만 개에서 재게 되어
#   (1) 두 표의 절대 수치를 나란히 놓을 수 없고,
#   (2) 4천 개는 정답 찾기가 너무 쉬워 before가 천장에 붙어 파인튜닝 개선폭(Δ)이 눌린다.
# 프로젝트 주 가설 H2(파인튜닝이 Recall·MRR을 올린다)를 숫자로 못 보여주게 되므로 맞춘다.
#
# 한계: run_compare는 로컬 축소 코퍼스가 아니라 HF 스트리밍(collect_corpus)으로 네거티브를
#   모은다. 그래서 "같은 20만 개"가 아니라 "같은 크기의 20만 개"까지만 맞출 수 있고,
#   스트리밍이 오래 걸린다. 완전히 동일한 인덱스로 맞추려면 run_compare가
#   data/ko_miracl_reduced_corpus.jsonl 을 읽어야 하는데 그건 rag/cli.py(파이프라인 영역)라
#   담당자 요청이 필요하다.
NEG_POOL_FULL = int(os.environ.get("NEG_POOL", 200_000))


def make_cfg() -> FinetuneConfig:
    if SMOKE:
        # 학습/평가/코퍼스를 소량으로 줄여 구조만 빠르게 확인 (숫자는 무의미)
        return FinetuneConfig(n_train_queries=30, n_eval_queries=10, neg_pool_size=300)
    # 검색·생성 평가와 같은 평가셋(dev 정답 쿼리 전수)을 쓰도록 개수를 맞춘다.
    dev_qrels = load_qrels(DATA, DATA.dev_split)
    n_eval = dev_qrels[dev_qrels[DATA.qr_score] > 0][DATA.qr_qid].nunique()
    return FinetuneConfig(n_eval_queries=n_eval, neg_pool_size=NEG_POOL_FULL)


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
    if not SMOKE and cfg.neg_pool_size < 100_000:
        print(f"경고: neg_pool={cfg.neg_pool_size} 는 검색 평가(20만)보다 작습니다 — "
              f"before가 부풀려져 파인튜닝 개선폭이 실제보다 작게 보입니다.")

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
