# ============================================
# Ko-miracl 기반 평가 실행 파일
# - 데이터 로딩(rag.data)과 지표 계산(rag.metrics)은 팀 파이프라인 걸 그대로 재사용
# - 코퍼스는 148만 개 전체를 스트리밍하는 대신, extraction/build_reduced_corpus.py로 미리 만든
#   data/ko_miracl_reduced_corpus.jsonl(BEIR 스타일 축소 코퍼스, 약 20만 청크)을 로컬 로드
# ============================================

import json
import os
import sys
from pathlib import Path

import chromadb
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.config import DATA, GenerationConfig
from rag.data import build_gold, load_qrels, load_queries, sample_pos_queries
from rag.embedding import load_embedder
from rag.index import build_collection, retrieve as rag_retrieve
from rag.metrics import evaluate

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "ko_miracl_reduced_corpus.jsonl"

# 스모크(SMOKE=1): 쿼리 20개 + 코퍼스 5천으로 "코드가 끝까지 도는지"만 확인(숫자는 무의미).
# 미설정(기본): 보고용 — 쿼리 전수 213개 + 코퍼스 전체(약 20만 청크).
#   * 213(질문 수)과 20만(검색 대상 문서 수)은 다른 축이다. 코퍼스가 작으면 정답 찾기가
#     쉬워져 검색 점수가 부풀려지므로, 보고용 숫자는 반드시 전체 코퍼스로 내야 한다.
SMOKE = bool(os.environ.get("SMOKE"))
SMOKE_N_EVAL = 20
SMOKE_CORPUS_LIMIT = 5000

cfg = GenerationConfig()

queries = load_queries(DATA)
dev_qrels = load_qrels(DATA, DATA.dev_split)

# 평가셋: dev split에서 정답 있는(score>0) 질문 "전수"(213개).
# 검색·생성·파인튜닝 비교가 모두 이 동일 평가셋을 쓴다(계획서 4.3 '동일 평가셋 고정').
n_available = dev_qrels[dev_qrels[DATA.qr_score] > 0][DATA.qr_qid].nunique()
eval_qids = sample_pos_queries(dev_qrels, DATA, n=SMOKE_N_EVAL if SMOKE else n_available)
gold = build_gold(dev_qrels, DATA, eval_qids)


def load_local_corpus(path: Path, limit=None, keep_ids=None):
    """코퍼스 로드. limit은 네거티브(비정답) 문서 상한(스모크용).

    keep_ids(정답 문서)는 limit과 무관하게 항상 포함한다 — 정답이 인덱스에 없으면
    그 쿼리는 애초에 맞출 수 없어 스모크조차 무의미해지기 때문.
    """
    corpus_text, corpus_title = {}, {}
    keep_ids = keep_ids or set()
    n_neg = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cid = row[DATA.c_id]
            if cid not in keep_ids:
                if limit is not None and n_neg >= limit:
                    continue
                n_neg += 1
            corpus_text[cid] = row[DATA.c_text]
            corpus_title[cid] = row[DATA.c_title]
    return corpus_text, corpus_title


if not CORPUS_PATH.exists():
    raise SystemExit(
        f"코퍼스 파일이 없습니다: {CORPUS_PATH}\n"
        "extraction/build_reduced_corpus.py 로 먼저 생성하거나, 저장소에서 받아오세요."
    )

gold_cids = {c for rels in gold.values() for c, s in rels.items() if s > 0}
corpus_text, corpus_title = load_local_corpus(
    CORPUS_PATH,
    limit=SMOKE_CORPUS_LIMIT if SMOKE else None,
    keep_ids=gold_cids,
)
index_cids = list(corpus_text.keys())

# 사전 점검: 정답 문서가 인덱스에 없으면 그 쿼리는 애초에 맞출 수 없다(지표가 조용히 떨어짐).
# 임베딩(수십 분) 전에 값싸게 확인해 둔다.
missing_gold = gold_cids - corpus_text.keys()
if missing_gold:
    print(f"경고: 정답 문서 {len(missing_gold)}/{len(gold_cids)}개가 코퍼스에 없습니다 — 상한이 그만큼 낮아집니다.")

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = load_embedder(cfg.embed_model, device, cfg.max_seq_len)

chroma_client = chromadb.Client()
collection = build_collection(
    chroma_client, "ko_miracl_eval", embed_model, index_cids, corpus_text, corpus_title,
    passage_prefix=cfg.passage_prefix, batch_size=cfg.batch_size,
)


def retrieve_fn(qid: str, k: int) -> list[str]:
    results = rag_retrieve(collection, embed_model, queries[qid], cfg.query_prefix, k=k)
    return [r["corpus_id"] for r in results]


if __name__ == "__main__":
    print("평가 쿼리 개수:", len(eval_qids), "(dev 정답 쿼리 전수) | 코퍼스 크기:", len(index_cids))
    # 계획서 4.3: 검색 지표는 k = 1 / 5 / 10 로 고정.
    # (cfg.top_k=5를 쓰면 k_list=[1,5,5]가 되어 @10이 빠지고 @5가 중복됨)
    result = evaluate(retrieve_fn, eval_qids, gold, k_list=[1, 5, 10], top_k=10)
    for name, value in result.items():
        print(f"{name}: {value:.4f}")
