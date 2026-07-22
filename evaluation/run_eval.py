# ============================================
# Ko-miracl 기반 평가 실행 파일
# - 데이터 로딩(rag.data)과 지표 계산(rag.metrics)은 팀 파이프라인 걸 그대로 재사용
# - retrieve_fn만 실제 구현(임베딩 + Chroma 인덱스)으로 교체하면 바로 동작함
# ============================================

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.config import DATA, GenerationConfig
from rag.data import build_gold, load_qrels, load_queries, sample_pos_queries
from rag.metrics import evaluate

cfg = GenerationConfig()

queries = load_queries(DATA)
dev_qrels = load_qrels(DATA, DATA.dev_split)

eval_qids = sample_pos_queries(dev_qrels, DATA, n=cfg.n_eval_queries)
gold = build_gold(dev_qrels, DATA, eval_qids)


def retrieve_fn(qid: str, k: int) -> list[str]:
    """
    TODO: 실제 retrieve로 교체
    - rag.data.collect_corpus 로 필요한 문서(정답 + 네거티브 풀) 텍스트 수집
    - rag.embedding.load_embedder 로 임베딩 모델 로드
    - rag.index.build_collection 으로 Chroma 컬렉션 인덱싱
    - rag.index.retrieve(collection, model, queries[qid], k=k) 결과에서
      [r["corpus_id"] for r in ...] 만 뽑아 반환하면 됨
    """
    raise NotImplementedError("retrieve_fn을 rag.embedding + rag.index 기반 실제 구현으로 교체하세요")


if __name__ == "__main__":
    print("평가 쿼리 개수:", len(eval_qids))
    result = evaluate(retrieve_fn, eval_qids, gold, k_list=[1, 5, cfg.top_k], top_k=cfg.top_k)
    for name, value in result.items():
        print(f"{name}: {value:.4f}")
