# ============================================
# Generation(RAG 답변) 품질 평가 — 3층 평가 구조 (계획서 4.3)
#   [1층] 규칙 기반 주 지표: 인용 정확성 / 환각율 / 기권 적절성  (rag/gen_metrics.py, 결정적)
#   [2층] 자동 의미 지표(RAGAS 등): 보류 — 외부 강판정기(GPT-4급) 확보 시 활성화.
#         로컬 판정기(Qwen 7B)는 신뢰성 낮음 + 파이프라인과 같은 모델이라 순환 편향 +
#         ragas/langchain 의존성 충돌 → 이 환경 미사용. 의미 품질 판정은 3층(사람)이 담당.
#   [3층] 사람 평가 시트 export (1·2층을 검증하는 앵커)
#
# 무근거 대조군(기권 적절성의 '근거 없음' 방향 측정용):
#   A) dev 정답 쿼리의 정답 문서를 인덱스에서 제외한 컬렉션 → 근거 없음(기권=정답). 정량 지표.
#   B) 코퍼스 밖 질문 몇 개 → 실사용 시연용 정성 예시(labels 불확실해 참고).
# ============================================

import os
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from transformers import set_seed

from rag.config import SEED
from rag.data import build_gold, needed_corpus_ids, sample_pos_queries
from rag.gen_metrics import ABSTENTION_MARKER, evaluate_generation
from rag.generation import rag_answer
from rag.index import build_collection

# run_retrieval_eval.py가 이미 만들어둔 코퍼스 로딩·임베딩·인덱스·모델을 그대로 재사용
# (코퍼스 임베딩을 중복으로 다시 안 함)
from evaluation.run_retrieval_eval import (
    DATA,
    cfg,
    chroma_client,
    collection,
    corpus_text,
    corpus_title,
    dev_qrels,
    embed_model,
    index_cids,
    queries,
)

# 스모크(SMOKE=1): 소량·단일 seed로 3층 구조만 빠르게 확인.
# 미설정(기본): 계획서 4.3 실제 설정(30개·seed 3개).
SMOKE = bool(os.environ.get("SMOKE"))

N_GEN_EVAL = 3 if SMOKE else 30    # answerable 평가 쿼리 수
N_UNANS = 3 if SMOKE else 30       # 무근거 대조군 A(정답 문서 제외) 쿼리 수
N_HUMAN_CHECK = 15                 # 사람이 같이 검산할 개수

# 무근거 대조군 B: 이 코퍼스(위키)에 답이 없는 질문 → strict 모드에서 기권이 정답.
# (라벨이 100% 확실치 않아 정량 주 지표가 아닌 '정성 예시'로만 사용)
OOD_QUESTIONS = [
    "오늘 서울 날씨 어때?",
    "내일 코스피 지수 얼마나 오를까?",
    "다음 주 로또 당첨 번호 알려줘",
    "가장 좋아하는 라면 브랜드 추천해줘",
    "이 문서 요약해서 이메일로 보내줘",
]

# dev score>0 쿼리를 seed=42 순서로 전부 뽑아, 겹치지 않게 분리:
#   앞 N_GEN_EVAL개 = answerable / 그 다음 N_UNANS개 = 무근거 대조군 A
all_pos_qids = sample_pos_queries(dev_qrels, DATA, n=10**9, seed=SEED)
gen_qids = all_pos_qids[:N_GEN_EVAL]
unans_qids = all_pos_qids[N_GEN_EVAL : N_GEN_EVAL + N_UNANS]

# 무근거 대조군 A용 인덱스: unans_qids의 정답 문서를 코퍼스에서 제외
unans_gold = build_gold(dev_qrels, DATA, unans_qids)
unans_gold_cids = needed_corpus_ids(unans_gold)
index_cids_no_gold = [c for c in index_cids if c not in unans_gold_cids]

def _answer_one(question: str, coll, seed: int = SEED) -> dict:
    # 생성이 API(GPT-4o-mini, temperature=0) 기반이라 응답은 사실상 결정적.
    # set_seed는 로컬 난수(검색/샘플링) 재현용으로만 남긴다.
    # seed 루프(GEN_SEEDS)는 API 호출 변동성 측정용으로 유지하되, temperature=0에서는
    # 답이 거의 같으므로 비용 아끼려면 GEN_SEEDS=[SEED]로 줄여도 된다.
    set_seed(seed)
    return rag_answer(
        question=question,
        collection=coll,
        embed_model=embed_model,
        query_prefix=cfg.query_prefix,
        k=cfg.top_k,
        mode="strict",
    )


def _to_record(set_name: str, answerable: bool, qid, question: str, result: dict) -> dict:
    return {
        "set": set_name,
        "answerable": answerable,
        "qid": qid,
        "question": question,
        "answer": result["answer"],
        "contexts": [c["text"] for c in result["contexts"]],
        "n_contexts": len(result["contexts"]),
    }


def build_coll_unans():
    # 무근거 A용 컬렉션: unans_qids의 정답 문서를 코퍼스에서 제외.
    # 임베딩은 seed와 무관(결정적)하므로 seed 루프 밖에서 한 번만 만들어 재사용한다.
    return build_collection(
        chroma_client,
        "ko_miracl_eval_unans",
        embed_model,
        index_cids_no_gold,
        corpus_text,
        corpus_title,
        passage_prefix=cfg.passage_prefix,
        batch_size=cfg.batch_size,
    )


def build_records(coll_unans, seed: int = SEED) -> list[dict]:
    records: list[dict] = []

    # answerable: 정답 문서가 인덱스에 그대로 있는 원 컬렉션에서 검색
    for qid in gen_qids:
        r = _answer_one(queries[qid], collection, seed)
        records.append(_to_record("answerable", True, qid, queries[qid], r))

    # 무근거 A: 정답 문서를 제외한 컬렉션 → 검색해도 근거가 없음(기권해야 정답)
    for qid in unans_qids:
        r = _answer_one(queries[qid], coll_unans, seed)
        records.append(_to_record("unanswerable_A", False, qid, queries[qid], r))

    # 무근거 B: 코퍼스 밖 질문 (원 컬렉션에서 검색 → 무관 문서만 상위에 옴)
    for q in OOD_QUESTIONS:
        r = _answer_one(q, collection, seed)
        records.append(_to_record("unanswerable_B", False, None, q, r))

    return records


# 계획서 4.3 신뢰성: seed 3개로 생성해 평균±표준편차 보고.
# 생성 LLM을 seed당 1회씩 전체(answerable+무근거) 돌리므로 seed 수만큼 느려진다.
# 스모크 테스트로 파이프라인만 확인할 땐 [SEED] 로 줄여라.
GEN_SEEDS = [SEED] if SMOKE else [SEED, SEED + 1, SEED + 2]


def mean_std(dicts: list[dict]) -> dict:
    """지표 dict 여러 개(=seed별 결과)를 받아 {metric: (평균, 표준편차)} 로 집계."""
    keys = dicts[0].keys()
    return {
        k: (float(np.mean([d[k] for d in dicts])), float(np.std([d[k] for d in dicts])))
        for k in keys
    }


# ---------- [2층] 자동 의미 지표 — 미구현(보류) ----------
# RAGAS 등 자동 의미 지표는 GPT-4급 '독립적·강한' 외부 판정기가 있어야 의미가 있다.
# 로컬 판정기(Qwen 7B)는 (1) 약한 판정으로 신뢰성 낮음 (2) 파이프라인과 같은 모델이라
# 순환 편향 (3) ragas/langchain 의존성 충돌 → 이 환경에서는 구현하지 않는다.
# 의미적 품질(정확성·근거 타당성)은 3층(사람 평가)이 담당.
# 외부 판정기(API) 확보 시: records0의 (question, answer, contexts)를 그대로 넘겨
# 별도 스크립트로 Faithfulness/Answer Relevancy/Context Precision을 계산하면 된다.


# ---------- [3층] 사람 평가 시트 ----------
def sample_human_check(records: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """사람 채점용 표본을 set별 층화 무작위로 뽑는다 (계획서 4.3 '무작위 표본').

    records[:n] 처럼 앞에서 자르면 answerable 만 뽑혀(gen_qids가 먼저 쌓임)
    환각율·기권 적절성(무근거 세트에서만 나오는 지표)을 사람이 검증할 수 없다.
    set(answerable / unanswerable_A / unanswerable_B)를 번갈아(라운드로빈) 뽑아
    모든 지표가 사람 대조를 받게 하고, seed를 고정해 재현/체리피킹 방지를 보장한다.
    """
    rng = random.Random(seed)
    by_set: dict[str, list[dict]] = {}
    for r in records:
        by_set.setdefault(r["set"], []).append(r)
    for rs in by_set.values():
        rng.shuffle(rs)

    picked: list[dict] = []
    sets = sorted(by_set)  # 결정적 순서
    cursor = {s: 0 for s in sets}
    while len(picked) < n and any(cursor[s] < len(by_set[s]) for s in sets):
        for s in sets:
            if cursor[s] < len(by_set[s]):
                picked.append(by_set[s][cursor[s]])
                cursor[s] += 1
                if len(picked) >= n:
                    break
    return picked


def export_human_check_sheet(records: list[dict], path: str = "evaluation/data/human_check.csv"):
    """사람이 채점할 표본을 CSV로 export (score 컬럼은 사람이 직접 채움)."""
    subset = sample_human_check(records, N_HUMAN_CHECK)
    df = pd.DataFrame(
        [
            {
                "set": r["set"],
                "qid": r["qid"],
                "question": r["question"],
                "answer": r["answer"],
                "answerable": r["answerable"],
                "correctness_score": "",     # 사람: 답변 정확성
                "groundedness_score": "",     # 사람: 근거 타당성
            }
            for r in subset
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("사람 채점용 시트 저장:", path)


if __name__ == "__main__":
    print(f"생성 평가: answerable {len(gen_qids)} / 무근거A {len(unans_qids)} / 무근거B {len(OOD_QUESTIONS)}")
    print(f"생성 seed: {GEN_SEEDS} (계획서 4.3 신뢰성: seed 평균±표준편차)")

    # 무근거 A 컬렉션은 seed와 무관(임베딩 결정적) → 한 번만 만들어 재사용
    coll_unans = build_coll_unans()

    # [1층] 규칙 기반 주 지표를 seed별로 계산 → 평균±표준편차.
    #  세트별 기권율·사람 시트는 첫 seed(records0) 답변을 대표로 사용.
    per_seed_scores: list[dict] = []
    records0: list[dict] = []
    for i, s in enumerate(GEN_SEEDS):
        print(f"\n----- 생성 (seed={s}) -----")
        recs = build_records(coll_unans, seed=s)
        per_seed_scores.append(evaluate_generation(recs))
        if i == 0:
            records0 = recs

    print("\n===== [1층] 규칙 기반 주 지표 (seed 평균 ± 표준편차) =====")
    for name, (mu, sd) in mean_std(per_seed_scores).items():
        print(f"{name}: {mu:.4f} ± {sd:.4f}")

    # 세트별 기권율 요약 (첫 seed 기준)
    print(f"\n----- 세트별 기권율 (seed={GEN_SEEDS[0]}) -----")
    df_rec = pd.DataFrame(records0)
    df_rec["abstained"] = df_rec["answer"].str.contains(ABSTENTION_MARKER)
    print(df_rec.groupby("set")["abstained"].mean().round(4))

    # [2층] 자동 의미 지표 — 보류. 사유·재활성화 방법은 위 '[2층]' 주석 블록 참고.
    print("\n===== [2층] 자동 의미 지표: 보류 — 의미 품질은 3층(사람 평가)이 담당 =====")

    # [3층] 사람 평가 시트 (무작위·set 층화 표본)
    print("\n===== [3층] 사람 평가 시트 =====")
    export_human_check_sheet(records0)
