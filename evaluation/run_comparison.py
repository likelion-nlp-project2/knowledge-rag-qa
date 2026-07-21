# ============================================
# chunk_size 비교 실행부
# - 250 / 300 / 500 / 800 을 각각 평가해서 hit@k, MRR을 한 표로 비교
# - 동점 규칙(README.md)에 따라 최적 chunk_size를 자동 결정
# - 진짜 retrieve 함수가 오면 chunk_size별 Chroma 컬렉션에 바인딩해 꽂기만 하면 됨
#   (retrieve 내부 구현과 분리 — 아래 retrieve_fns 딕셔너리만 교체)
# ============================================

from test_queries import test_queries_retrieval
from evaluate import evaluate_retrieval, evaluate_mrr

# RAG가 LLM에 실제로 넣는 chunk 개수와 동일하게 맞출 것 (미정 → 잠정 5)
K = 5

CHUNK_SIZES = [250, 300, 500, 800]


def compare(retrieve_fns, k=K):
    """
    retrieve_fns: {chunk_size: retrieve_fn}
      - retrieve_fn(query) -> [{"text","score","metadata":{"doc_id",...}}, ...]
      - 진짜 retrieve를 chunk_size별 컬렉션에 바인딩해서 넣으면 됨
    반환: 동점 규칙으로 정렬된 결과 리스트 (0번이 최적)
    """
    rows = []
    for cs, fn in retrieve_fns.items():
        rows.append({
            "chunk_size": cs,
            "hit": evaluate_retrieval(fn, test_queries_retrieval, k=k),
            "mrr": evaluate_mrr(fn, test_queries_retrieval),
        })
    # 동점 규칙: 1) hit@k 높은 순  2) MRR 높은 순  3) chunk_size 작은 순
    return sorted(rows, key=lambda r: (-r["hit"], -r["mrr"], r["chunk_size"]))


def print_table(ranked, k=K):
    print(f"{'chunk_size':>10} | {'hit@'+str(k):>8} | {'MRR':>6}")
    print("-" * 32)
    for r in ranked:
        print(f"{r['chunk_size']:>10} | {r['hit']:>8.3f} | {r['mrr']:>6.3f}")
    best = ranked[0]
    print(f"\n>>> 최적 chunk_size: {best['chunk_size']} "
          f"(hit@{k}={best['hit']:.3f}, MRR={best['mrr']:.3f})")


if __name__ == "__main__":
    from mock_functions import retrieve
    # TODO: 진짜 retrieve 완성되면 chunk_size별 Chroma 컬렉션에 바인딩해서 교체
    #   예) retrieve_fns = {cs: make_retrieve(collection=f"chunks_{cs}") for cs in CHUNK_SIZES}
    retrieve_fns = {cs: retrieve for cs in CHUNK_SIZES}  # 지금은 mock (전부 동일 → 검증용)
    print_table(compare(retrieve_fns))
