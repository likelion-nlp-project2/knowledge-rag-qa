# ============================================
# 리랭커 비교: none / bge-reranker-v2-m3(기본) / bge-reranker-base / ko-reranker
# - 배포된 검색 API(SEARCH_API_URL)를 그대로 호출, 리랭커만 바꿔가며 같은 평가셋으로 비교
# - 원 질의 그대로 검색(LLM 호출 없음) — HyDE 비교보다 빠르고 비용 없음
# ============================================

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import requests

from rag.config import DATA
from rag.data import build_gold, load_qrels, load_queries, sample_pos_queries
from rag.metrics import evaluate
from rag_engine import SEARCH_API_URL

N_EVAL = int(os.environ.get("N_EVAL", 50))
K_LIST = [1, 5, 10]
TOP_K = 10
RERANKERS = ["none", "bge-reranker-v2-m3", "bge-reranker-base", "ko-reranker"]

print("쿼리/정답(qrels) 로딩 중 (taeminlee/Ko-miracl, HF)...")
queries = load_queries(DATA)
dev_qrels = load_qrels(DATA, DATA.dev_split)
eval_qids = sample_pos_queries(dev_qrels, DATA, n=N_EVAL)
gold = build_gold(dev_qrels, DATA, eval_qids)


def search(query: str, rerank: str, k: int) -> list:
    r = requests.post(
        f"{SEARCH_API_URL}/search",
        json={"query": query, "k": k, "rerank": rerank},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    ranked: list = []
    for res in data["results"]:
        for cid in res["metadata"]["matched_chunks"].split("|"):
            if cid and cid not in ranked:
                ranked.append(cid)
    return ranked


def make_retrieve_fn(rerank: str):
    def _retrieve(qid: str, k: int):
        return search(queries[qid], rerank, k)[:k]

    return _retrieve


if __name__ == "__main__":
    print(f"평가 쿼리 {len(eval_qids)}개 | top_k={TOP_K} | 리랭커 {len(RERANKERS)}개 비교")

    results = {}
    for i, rerank in enumerate(RERANKERS, 1):
        print(f"[{i}/{len(RERANKERS)}] {rerank} 평가 중...")
        results[rerank] = evaluate(make_retrieve_fn(rerank), eval_qids, gold, K_LIST, TOP_K)

    out_path = Path(__file__).resolve().parent / "data" / "reranker_compare_result.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps({"n_eval": len(eval_qids), "top_k": TOP_K, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    header = "metric".ljust(10) + "".join(r.rjust(20) for r in RERANKERS)
    print("\n" + header)
    for m in results[RERANKERS[0]]:
        row = m.ljust(10) + "".join(f"{results[r][m]:20.4f}" for r in RERANKERS)
        print(row)
    print(f"\n결과 저장: {out_path}")
