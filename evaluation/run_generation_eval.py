# ============================================
# Generation(RAG 답변) 품질 평가 실행 파일
# - retrieval 평가(run_retrieval_eval.py)와 같은 dev 쿼리 풀에서 30개만 서브샘플
# - 판정 LLM은 일단 팀이 이미 로컬로 띄우는 Qwen2.5-7B-Instruct 재사용 (API 비용 없음)
# - Ko-miracl qrels에는 정답 "문서 id"만 있고 정답 "답변 텍스트"가 없어서,
#   ground_truth가 필요한 Context Recall은 제외
# => Faithfulness/Answer Relevancy/Context Precision 3개만 사용
# ============================================

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datasets import Dataset

from rag.data import sample_pos_queries
from rag.generation import rag_answer
from rag.llm import load_llm

# run_retrieval_eval.py가 이미 만들어둔 코퍼스 임베딩/인덱스(collection, embed_model)와
# queries/dev_qrels/cfg를 그대로 재사용 (코퍼스 로딩·임베딩을 중복으로 다시 안 함)
from evaluation.run_retrieval_eval import DATA, cfg, collection, dev_qrels, embed_model, queries

N_GEN_EVAL = 30
N_HUMAN_CHECK = 15  # N_GEN_EVAL 중 사람이 같이 검산할 개수

# retrieval 평가(run_retrieval_eval.py)와 동일한 seed=42 셔플 순서를 사용해서,
# 여기서 뽑는 30개는 항상 그 213개 dev 평가셋의 앞부분 서브셋과 일치함
gen_qids = sample_pos_queries(dev_qrels, DATA, n=N_GEN_EVAL)

tok, llm = load_llm(cfg.llm_name)


def build_rag_answers() -> list[dict]:
    rows = []
    for qid in gen_qids:
        result = rag_answer(
            question=queries[qid],
            collection=collection,
            embed_model=embed_model,
            tok=tok,
            llm=llm,
            query_prefix=cfg.query_prefix,
            k=cfg.top_k,
        )
        rows.append({
            "qid": qid,
            "question": result["question"],
            "answer": result["answer"],
            "contexts": [c["text"] for c in result["contexts"]],
        })
    return rows


def run_ragas(rows: list[dict]) -> pd.DataFrame:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
    from ragas import evaluate as ragas_evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
    from transformers import pipeline as hf_pipeline

    # 생성에 쓴 것과 같은 tok/llm을 그대로 판정용으로 재사용 (별도 로드 안 함)
    gen_pipeline = hf_pipeline(
        "text-generation", model=llm, tokenizer=tok, max_new_tokens=512,
    )
    judge_llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=gen_pipeline))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=cfg.embed_model))

    dataset = Dataset.from_list(rows)
    result = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    return result.to_pandas()


def export_human_check_sheet(rows: list[dict], path: str = "evaluation/data/human_check.csv"):
    """사람이 채점할 앞 N_HUMAN_CHECK개를 CSV로 뽑아둠 (score 컬럼은 사람이 직접 채움)"""
    subset = rows[:N_HUMAN_CHECK]
    df = pd.DataFrame([{
        "qid": r["qid"],
        "question": r["question"],
        "answer": r["answer"],
        "faithfulness_score": "",
        "relevancy_score": "",
    } for r in subset])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("사람 채점용 시트 저장:", path)


if __name__ == "__main__":
    print("Generation 평가 쿼리 개수:", len(gen_qids))
    rows = build_rag_answers()
    ragas_result = run_ragas(rows)
    print(ragas_result)
    export_human_check_sheet(rows)
