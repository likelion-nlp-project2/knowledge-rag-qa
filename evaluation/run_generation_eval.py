# ============================================
# Generation(RAG 답변) 품질 평가 — 자동(LLM) 평가 (계획서 4.3)
#   [1층] 규칙 기반 주 지표: 인용 정확성 / 환각율 / 기권 적절성  (rag/gen_metrics.py, 결정적)
#   [2층] RAGAS 의미 지표: faithfulness / answer_relevancy (판정기 API, temperature=0)
#         판정기가 생성 모델(gpt-4o-mini)보다 상위라야 채점이 의미를 가진다(멘토 피드백 #5).
#   * 사람 평가는 팀 결정으로 제외(자동 LLM 평가로만 수행). 이전 구현은 git 이력 참고.
#
# 필요 패키지: ragas>=0.4, langchain-openai   /  필요 환경변수: OPENAI_API_KEY
#
# 무근거 대조군(기권 적절성의 '근거 없음' 방향 측정용):
#   A) dev 정답 쿼리의 정답 문서를 인덱스에서 제외한 컬렉션 → 근거 없음(기권=정답). 정량 지표.
#   B) 코퍼스 밖 질문 몇 개 → 실사용 시연용 정성 예시(labels 불확실해 참고).
#      B는 라벨이 불확실하므로 정량 집계에서 제외하고 따로 보고한다(아래 QUANT_SETS).
# ============================================

import json
import os
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


# 무근거 대조군 B: 이 코퍼스(위키)에 답이 없는 질문 → strict 모드에서 기권이 정답.
# (라벨이 100% 확실치 않아 정량 주 지표가 아닌 '정성 예시'로만 사용)
OOD_QUESTIONS = [
    "오늘 서울 날씨 어때?",
    "내일 코스피 지수 얼마나 오를까?",
    "다음 주 로또 당첨 번호 알려줘",
    "가장 좋아하는 라면 브랜드 추천해줘",
    "이 문서 요약해서 이메일로 보내줘",
]

# 평가셋: dev score>0 쿼리 "전수"(213개)를 seed=42 순서로 고정해 사용.
# 팀 피드백("규모 213개 유지")에 따라 예전처럼 앞 30개/다음 30개로 쪼개지 않는다.
# 쪼개면 각 조건이 절반으로 줄어 213 규모를 못 지키므로, 같은 213개를 두 조건으로 각각 돌린다:
#   answerable     = 정답 문서가 인덱스에 있는 원 컬렉션  → 기권하면 과잉 기권(나쁨)
#   unanswerable_A = 그 213개의 정답 문서를 뺀 컬렉션      → 기권해야 정답
# 스모크(SMOKE=1): 쿼리 3개로 "코드가 끝까지 도는지"만 확인(숫자는 무의미).
#   RAGAS 판정이 유료 API라, 전량 실행 전에 반드시 이걸로 구조를 먼저 검증할 것.
# 미설정(기본): 보고용 — dev 정답 쿼리 전수 213개.
#
# 평가에 train split을 절대 섞지 않는다: train은 리트리버 파인튜닝 학습에 쓰이므로
# 평가에 넣으면 데이터 누수가 된다(학습에 본 질문으로 성능을 재는 셈).
SMOKE = bool(os.environ.get("SMOKE"))

all_pos_qids = sample_pos_queries(dev_qrels, DATA, n=10**9, seed=SEED)
eval_qids = all_pos_qids[:3] if SMOKE else all_pos_qids

# 무근거 대조군 A용 인덱스: eval_qids 전체의 정답 문서를 코퍼스에서 제외
unans_gold = build_gold(dev_qrels, DATA, eval_qids)
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
        mode="fewshot",  # 도현님 프롬프트 A/B 실험 결과 채택 (faithfulness/answer_relevancy 둘 다 우위)
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
    # 무근거 A용 컬렉션: eval_qids 전체의 정답 문서를 코퍼스에서 제외.
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


# 정량 주 지표에 넣을 세트. 무근거 B(OOD)는 "이 코퍼스에 답이 없다"는 라벨이 우리 추정이라
# 정량 집계에 넣으면 환각율·기권 적절성이 검증 안 된 라벨로 오염된다. 따로 정성 보고만 한다.
QUANT_SETS = ("answerable", "unanswerable_A")


def build_records(coll_unans, seed: int = SEED) -> list[dict]:
    records: list[dict] = []

    # answerable: 정답 문서가 인덱스에 그대로 있는 원 컬렉션에서 검색
    for qid in eval_qids:
        r = _answer_one(queries[qid], collection, seed)
        records.append(_to_record("answerable", True, qid, queries[qid], r))

    # 무근거 A: 같은 쿼리를 정답 문서 제외 컬렉션으로 → 근거가 없음(기권해야 정답)
    for qid in eval_qids:
        r = _answer_one(queries[qid], coll_unans, seed)
        records.append(_to_record("unanswerable_A", False, qid, queries[qid], r))

    # 무근거 B: 코퍼스 밖 질문 (원 컬렉션에서 검색 → 무관 문서만 상위에 옴)
    for q in OOD_QUESTIONS:
        r = _answer_one(q, collection, seed)
        records.append(_to_record("unanswerable_B", False, None, q, r))

    return records


# 계획서 4.3 신뢰성(seed 반복) — 생성은 1회로 충분하다.
#
# 예전엔 생성이 로컬 Qwen + temperature=0.2 라 같은 입력에도 답이 흔들려 seed 3개가 필요했다.
# 지금은 생성이 API(gpt-4o-mini) + temperature=0 이라 사실상 결정적이라, 반복해도
# 표준편차가 0에 수렴한다. 3회로 두면 같은 답을 세 번 받으려고 RAGAS 판정비를 3배 낼 뿐이다.
#
# 단, 파인튜닝 전/후 비교(run_finetune_compare_eval.py)는 매번 새로 학습해 실제로 변동이
# 생기므로 거기서는 seed 3개를 그대로 유지한다 — 멘토 피드백 #1의 표준편차는 그쪽이 담당.
#
# 생성 변동성을 굳이 확인하고 싶으면 GEN_SEEDS 에 seed를 더 넣으면 된다(비용 비례 증가).
GEN_SEEDS = [SEED]


def mean_std(dicts: list[dict]) -> dict:
    """지표 dict 여러 개(=seed별 결과)를 받아 {metric: (평균, 표준편차)} 로 집계."""
    keys = dicts[0].keys()
    return {
        k: (float(np.mean([d[k] for d in dicts])), float(np.std([d[k] for d in dicts])))
        for k in keys
    }


# ---------- [2층] RAGAS 자동 의미 지표 ----------
# 사람 평가를 폐기하고 자동 LLM 판정으로 전환(팀 피드백)했으므로, 이 층이 생성 '의미 품질'의
# 사실상 주 지표가 된다. 그래서 1층과 동일하게 seed별로 판정해 평균±표준편차까지 보고한다.
#
# 판정기(temperature=0). 생성 모델(gpt-4o-mini)보다 상위여야 채점이 의미를 가진다(피드백 #5).
# 지표: faithfulness(근거 충실도) + answer_relevancy(답변 관련성).
#   context_precision은 RAGAS가 ground_truth(정답 '문장')를 요구하는데, Ko-miracl에는
#   정답 '문서'만 있고 정답 답변 텍스트가 없어 사용하지 않는다.
# 판정 대상: answerable 세트에서 기권하지 않은 답변만.
#   (기권 답변의 품질은 1층의 over_abstention_rate가 담당 — 여기서 이중으로 벌점 주지 않음)
#
# 판정기는 evaluation/run_ragas_eval.py(프롬프트 A/B 실험)와 반드시 같아야 한다.
#
# 두 스크립트는 목적이 달라 트랙을 따로 유지하지만(이쪽은 계획서 4.3 보고용 절대 수치,
# 저쪽은 프롬프트 A/B 상대 비교), 둘 다 'faithfulness'라는 같은 이름의 지표를 낸다.
# 판정 모델이 다르면 점수 스케일이 달라져, 발표에서 두 숫자가 나란히 보이는 순간 사고가 난다.
# 그래서 평가셋·검색 백엔드는 달라도 되지만 판정기만은 하나로 고정한다.
#
# 기본값을 gpt-5.4-mini 로 둔 이유: (1) run_ragas_eval.py 가 이미 이 모델이고, (2) 생성
# 모델 gpt-4o-mini 와 계열이 달라 자기 채점 편향에서 더 자유롭다(멘토 피드백 #5).
# 팀이 gpt-4o 로 정하면 이 한 줄만 바꾸면 된다(환경변수로도 덮어쓸 수 있음).
RAGAS_JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "gpt-5.4-mini")
RAGAS_EMBED_MODEL = "text-embedding-3-small"   # answer_relevancy 계산에 필요
JUDGE_CSV = "result/ragas_scores.csv"
JUDGE_COLS = ["faithfulness", "answer_relevancy"]

# 정답 답변 라벨(RAGAS 의 reference = ground_truth). evaluation/build_reference_answers.py 산출물.
# 있으면 4컬럼 지표(context_recall / answer_correctness)가 자동으로 추가되고, 없으면
# 3컬럼 지표만 낸다 — 라벨 없이도 파이프라인이 그대로 돌아간다.
REFERENCE_JSONL = "result/reference_answers.jsonl"
REF_COLS = ["context_recall", "answer_correctness"]

# 생성 답변 원본. Colab 세션이 끊기면 출력이 사라지는데 이건 API 비용이 든 산출물이라
# 판정(RAGAS)보다 먼저 디스크에 떨군다 — 판정이 실패해도 재생성 없이 다시 채점할 수 있다.
RECORDS_JSONL = "result/generations_eval.jsonl"


def _save_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _build_ragas_judge():
    """RAGAS 판정기(LLM + 임베딩) 구성. OPENAI_API_KEY 환경변수 필요(코드에 키를 넣지 말 것)."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            f"OPENAI_API_KEY 환경변수가 없습니다. RAGAS 판정기는 {RAGAS_JUDGE_MODEL}를 사용합니다.\n"
            "  예) .env 에 OPENAI_API_KEY=... (rag/llm.py가 load_dotenv로 읽는 파일과 동일)"
        )
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # temperature=0: 판정 재현성(계획서 4.3 신뢰성)
    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=RAGAS_JUDGE_MODEL, temperature=0))
    judge_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=RAGAS_EMBED_MODEL))
    return judge_llm, judge_emb


def load_references() -> dict:
    """qid -> 정답 답변 텍스트. 라벨 파일이 없으면 빈 dict."""
    p = Path(REFERENCE_JSONL)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # reference=None 은 '정답 문서에 답이 없어 라벨을 못 만든 질문'이라 집계에서 뺀다
    return {r["qid"]: r["reference"] for r in rows if r.get("reference")}


def run_ragas(records: list[dict], seed: int) -> pd.DataFrame:
    """answerable·비기권 답변을 RAGAS로 판정해 seed 컬럼을 붙인 점수표를 반환.

    RAGAS 0.4 API 기준. 0.1 시절의 `from ragas.metrics import faithfulness`(모듈 레벨
    인스턴스)와 HF Dataset 입력은 0.4에서 제거됐고, EvaluationDataset + 클래스형 지표로
    바뀌었다. 필드명도 question/answer/contexts -> user_input/response/retrieved_contexts,
    ground_truth -> reference 로 바뀌었다(구버전 예제 코드를 그대로 붙이면 안 됨).

    지표를 두 번에 나눠 돌린다:
      3컬럼(faithfulness/answer_relevancy) — reference 불필요 → 대상 '전체'
      4컬럼(context_recall/answer_correctness) — reference 필요 → 라벨 있는 것만
    한 번에 돌리면 라벨 없는 질문 때문에 3컬럼 지표의 모수까지 같이 줄어들어,
    라벨 도입 전 실행(result/ragas_summary.csv)과 숫자를 비교할 수 없게 된다.
    """
    from ragas import EvaluationDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import AnswerCorrectness, Faithfulness, LLMContextRecall, ResponseRelevancy

    targets = [
        r for r in records
        if r["answerable"] and ABSTENTION_MARKER not in r["answer"]
    ]
    if not targets:
        return pd.DataFrame()

    judge_llm, judge_emb = _build_ragas_judge()

    def _run(rows: list[dict], samples: list[dict], metrics) -> pd.DataFrame:
        out = ragas_evaluate(EvaluationDataset.from_list(samples), metrics=metrics).to_pandas()
        out.insert(0, "qid", [r["qid"] for r in rows])
        return out

    df = _run(
        targets,
        [{"user_input": r["question"], "response": r["answer"],
          "retrieved_contexts": r["contexts"]} for r in targets],
        [Faithfulness(llm=judge_llm), ResponseRelevancy(llm=judge_llm, embeddings=judge_emb)],
    )

    # ── 4컬럼 지표: 정답 답변 라벨이 있는 질문만 ──
    refs = load_references()
    ref_targets = [r for r in targets if refs.get(r["qid"])]
    if ref_targets:
        print(f"  4컬럼 지표: {len(ref_targets)}/{len(targets)}건에 정답 라벨 있음")
        ref_df = _run(
            ref_targets,
            [{"user_input": r["question"], "response": r["answer"],
              "retrieved_contexts": r["contexts"], "reference": refs[r["qid"]]}
             for r in ref_targets],
            [LLMContextRecall(llm=judge_llm),
             AnswerCorrectness(llm=judge_llm, embeddings=judge_emb)],
        )
        keep = ["qid"] + [c for c in REF_COLS if c in ref_df.columns]
        df = df.merge(ref_df[keep], on="qid", how="left")
    else:
        print(f"  4컬럼 지표: 건너뜀 (정답 라벨 없음 — {REFERENCE_JSONL})")

    df.insert(0, "seed", seed)
    return df


def _report(label: str, per_seed: list[dict]) -> None:
    """seed가 1개면 값만, 여러 개면 평균±표준편차로 출력."""
    if len(per_seed) == 1:
        for name, value in per_seed[0].items():
            print(f"{name}: {value:.4f}")
        print(f"  ({label}: 생성 temperature=0 이라 결정적 — seed 반복 불필요)")
    else:
        for name, (mu, sd) in mean_std(per_seed).items():
            print(f"{name}: {mu:.4f} ± {sd:.4f}")


if __name__ == "__main__":
    # 같은 eval_qids를 두 조건(정답 포함/제외)으로 돌리므로 생성 건수는 2×213 + OOD
    print(f"생성 평가: 평가셋 {len(eval_qids)}개 × 2조건(answerable/무근거A) + 무근거B {len(OOD_QUESTIONS)}")
    print(f"평가 split: dev only (train은 파인튜닝 학습에 쓰이므로 누수 방지 차원에서 제외)")
    print(f"생성 seed: {GEN_SEEDS} | RAGAS 판정기: {RAGAS_JUDGE_MODEL}")
    _refs = load_references()
    print(f"정답 라벨: {len(_refs)}건 ({REFERENCE_JSONL})"
          if _refs else
          f"정답 라벨 없음 → 3컬럼 지표만 계산. 만들려면: python evaluation/build_reference_answers.py")

    # 무근거 A 컬렉션은 seed와 무관(임베딩 결정적) → 한 번만 만들어 재사용
    coll_unans = build_coll_unans()

    per_seed_scores: list[dict] = []
    per_seed_judge: list[dict] = []
    judge_frames: list[pd.DataFrame] = []
    records0: list[dict] = []
    for i, s in enumerate(GEN_SEEDS):
        print(f"\n----- 생성·판정 (seed={s}) -----")
        recs = build_records(coll_unans, seed=s)
        if i == 0:
            records0 = recs
            _save_jsonl(RECORDS_JSONL, recs)   # 판정 전에 저장 — 판정이 죽어도 생성분은 지킨다
            print(f"생성 {len(recs)}건 저장: {RECORDS_JSONL}")

        # 정량 지표는 라벨이 확실한 세트만 (무근거 B 제외)
        per_seed_scores.append(evaluate_generation([r for r in recs if r["set"] in QUANT_SETS]))

        jdf = run_ragas(recs, s)
        judge_frames.append(jdf)
        if len(jdf):
            # 4컬럼 지표는 라벨이 있을 때만 컬럼이 생긴다 → 있는 것만 집계
            cols = [c for c in JUDGE_COLS + REF_COLS if c in jdf.columns]
            per_seed_judge.append(jdf[cols].mean().to_dict())

    print(f"\n===== [1층] 규칙 기반 주 지표 (대상: {', '.join(QUANT_SETS)}) =====")
    _report("1층", per_seed_scores)

    # 세트별 기권율 요약 (첫 seed 기준) — 무근거 B도 여기서는 참고로 함께 보여준다
    print(f"\n----- 세트별 기권율 (seed={GEN_SEEDS[0]}) -----")
    df_rec = pd.DataFrame(records0)
    df_rec["abstained"] = df_rec["answer"].str.contains(ABSTENTION_MARKER)
    print(df_rec.groupby("set")["abstained"].mean().round(4))

    # 무근거 B는 정성 예시로만 — 실제 답변을 눈으로 확인한다
    print("\n----- [정성] 무근거 B(코퍼스 밖 질문) 응답 예시 -----")
    for r in records0:
        if r["set"] == "unanswerable_B":
            mark = "기권" if ABSTENTION_MARKER in r["answer"] else "답변함"
            print(f"  [{mark}] {r['question']}\n         -> {r['answer'][:80]}")

    # [2층] RAGAS 의미 지표 — answerable·비기권 답변만
    print(f"\n===== [2층] RAGAS 의미 지표 (판정기 {RAGAS_JUDGE_MODEL}, 0~1) =====")
    if per_seed_judge:
        _report("2층", per_seed_judge)
        all_judge = pd.concat(judge_frames, ignore_index=True)
        # 4컬럼 지표의 NaN은 '라벨이 없어서 안 잰 것'이라 판정 실패가 아니다 → 3컬럼만 센다
        n_fail = int(all_judge[JUDGE_COLS].isna().sum().sum())
        Path(JUDGE_CSV).parent.mkdir(parents=True, exist_ok=True)
        all_judge.to_csv(JUDGE_CSV, index=False, encoding="utf-8-sig")
        print(f"판정 {len(all_judge)}건 저장: {JUDGE_CSV} (점수 누락 {n_fail}건)")
    else:
        print("판정 대상 없음 (answerable 답변이 모두 기권)")
