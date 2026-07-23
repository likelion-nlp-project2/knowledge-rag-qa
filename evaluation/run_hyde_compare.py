# ============================================
# HyDE 질의 재작성 전/후 검색 품질 비교 (Hit/MRR/nDCG)
# - 로컬 임베딩 인덱스 없이, 이미 배포된 검색 API(SEARCH_API_URL)를 그대로 호출한다.
# - 베이스라인: 원 질의 그대로 검색 / HyDE: 가상 답변 문단을 LLM으로 생성해 검색
# - dense 단계 자체의 효과를 보려고 rerank=none 으로 비교한다(리랭커가 차이를 가릴 수 있어서).
# ============================================

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.config import DATA
from rag.data import build_gold, load_qrels, load_queries, sample_pos_queries
from rag.metrics import evaluate
from rag_engine import _search_api, hyde_rewrite
from schema import PipelineConfig

N_EVAL = int(os.environ.get("N_EVAL", 50))
K_LIST = [1, 5, 10]
TOP_K = 10

cfg = PipelineConfig(top_k_retrieval=TOP_K, use_reranker=False)  # dense-only 비교

print("쿼리/정답(qrels) 로딩 중 (taeminlee/Ko-miracl, HF)...")
queries = load_queries(DATA)
dev_qrels = load_qrels(DATA, DATA.dev_split)
eval_qids = sample_pos_queries(dev_qrels, DATA, n=N_EVAL)
gold = build_gold(dev_qrels, DATA, eval_qids)

hyde_cache: dict = {}


def make_retrieve_fn(use_hyde: bool):
    def _retrieve(qid: str, k: int):
        q = queries[qid]
        if use_hyde:
            if qid not in hyde_cache:
                hyde_cache[qid] = hyde_rewrite(q)
            q = hyde_cache[qid]
        chunks = _search_api(q, cfg)
        ranked: list = []
        for c in chunks:
            for cid in c.metadata.matched_chunks.split("|"):
                if cid and cid not in ranked:
                    ranked.append(cid)
        return ranked[:k]

    return _retrieve


if __name__ == "__main__":
    print(f"평가 쿼리 {len(eval_qids)}개 | dense-only(rerank=none) | top_k={TOP_K}")

    print("\n[1/2] 베이스라인(원 질의) 평가 중...")
    baseline = evaluate(make_retrieve_fn(False), eval_qids, gold, K_LIST, TOP_K)

    print("[2/2] HyDE 평가 중 (쿼리당 LLM 호출 1회 추가)...")
    hyde = evaluate(make_retrieve_fn(True), eval_qids, gold, K_LIST, TOP_K)

    result = {
        "n_eval": len(eval_qids),
        "top_k": TOP_K,
        "baseline": baseline,
        "hyde": hyde,
        "hyde_samples": [
            {"qid": qid, "question": queries[qid], "hyde_doc": hyde_cache[qid]}
            for qid in eval_qids[:5]
            if qid in hyde_cache
        ],
    }
    out_path = Path(__file__).resolve().parent / "data" / "hyde_compare_result.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'metric':10s} {'baseline':>10s} {'hyde':>10s} {'Δ':>10s}")
    for m in baseline:
        b, h = baseline[m], hyde[m]
        print(f"{m:10s} {b:10.4f} {h:10.4f} {h - b:+10.4f}")
    print(f"\n결과 저장: {out_path}")
