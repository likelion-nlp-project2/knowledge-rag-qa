# ============================================
# dense 후보 풀 크기(n) 비교
# - 가설: 질문 표현이 코퍼스 문서와 다르면, 정답 문서가 애초에 리랭크 전 후보(top-n)에
#   못 들어와서 리랭커가 아무리 좋아도 못 살린다. n을 넓히면 이게 나아지는지 확인.
# - 리랭커는 고정(bge-reranker-v2-m3, 앱 실제 기본값), n만 바꿔가며 비교.
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
RERANK = "bge-reranker-v2-m3"
POOL_SIZES = [50, 100, 200]  # 50 = 현재 기본값(INFRA.retrieve_n)

print("쿼리/정답(qrels) 로딩 중 (taeminlee/Ko-miracl, HF)...")
queries = load_queries(DATA)
dev_qrels = load_qrels(DATA, DATA.dev_split)
eval_qids = sample_pos_queries(dev_qrels, DATA, n=N_EVAL)
gold = build_gold(dev_qrels, DATA, eval_qids)


def search(query: str, n: int, k: int) -> list:
    r = requests.post(
        f"{SEARCH_API_URL}/search",
        json={"query": query, "k": k, "n": n, "rerank": RERANK},
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


def make_retrieve_fn(n: int):
    def _retrieve(qid: str, k: int):
        return search(queries[qid], n, k)[:k]

    return _retrieve


if __name__ == "__main__":
    print(f"평가 쿼리 {len(eval_qids)}개 | top_k={TOP_K} | rerank={RERANK} | n 후보 {POOL_SIZES}")

    results = {}
    for i, n in enumerate(POOL_SIZES, 1):
        print(f"[{i}/{len(POOL_SIZES)}] n={n} 평가 중...")
        results[str(n)] = evaluate(make_retrieve_fn(n), eval_qids, gold, K_LIST, TOP_K)

    out_path = Path(__file__).resolve().parent / "data" / "pool_compare_result.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps({"n_eval": len(eval_qids), "top_k": TOP_K, "rerank": RERANK, "results": results},
                    ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    header = "metric".ljust(10) + "".join(f"n={n}".rjust(14) for n in POOL_SIZES)
    print("\n" + header)
    for m in results[str(POOL_SIZES[0])]:
        row = m.ljust(10) + "".join(f"{results[str(n)][m]:14.4f}" for n in POOL_SIZES)
        print(row)
    print(f"\n결과 저장: {out_path}")
