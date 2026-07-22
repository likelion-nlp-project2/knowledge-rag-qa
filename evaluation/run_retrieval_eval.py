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
from rag.data import build_gold, collect_corpus, load_qrels, load_queries, sample_pos_queries
from rag.embedding import load_embedder
from rag.index import build_collection, retrieve as rag_retrieve
from rag.metrics import evaluate

# 스모크(SMOKE=1): 5천 subset 파일(ko_miracl_subset.jsonl)로 빠르게 3층 구조만 확인.
# 미설정(기본): reduced corpus 전체(약 20만) = 보고용 베이스라인.
SMOKE = bool(os.environ.get("SMOKE"))
SMOKE_CORPUS_LIMIT = 5000
CORPUS_FILE = "ko_miracl_subset.jsonl" if SMOKE else "ko_miracl_reduced_corpus.jsonl"
CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / CORPUS_FILE

cfg = GenerationConfig()

queries = load_queries(DATA)
dev_qrels = load_qrels(DATA, DATA.dev_split)

# dev split에서 정답 있는(score>0) 질문. 스모크는 소량(20)만 → 코퍼스 스트리밍 시간 단축.
n_available = dev_qrels[dev_qrels[DATA.qr_score] > 0][DATA.qr_qid].nunique()
N_EVAL = 20 if SMOKE else n_available
eval_qids = sample_pos_queries(dev_qrels, DATA, n=N_EVAL)
gold = build_gold(dev_qrels, DATA, eval_qids)


def load_local_corpus(path, limit=None, keep_ids=None):
    # limit: 네거티브(비정답) 문서 최대 개수(스모크용). keep_ids(정답 문서)는 limit과 무관하게 항상 포함.
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


# 정답 문서는 스모크에서도 반드시 인덱스에 있어야 answerable 쿼리가 의미를 가짐
gold_cids = {c for rels in gold.values() for c, s in rels.items() if s > 0}
if SMOKE:
    # 로컬 파일에 의존하지 않고 HF에서 정답+네거티브만 스트리밍(몇 분 소요).
    # VM이 재활용돼 data/*.jsonl 이 없어도 동작한다.
    corpus_text, corpus_title = collect_corpus(DATA, gold_cids, SMOKE_CORPUS_LIMIT)
else:
    corpus_text, corpus_title = load_local_corpus(CORPUS_PATH, keep_ids=gold_cids)
index_cids = list(corpus_text.keys())

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
    print("평가 쿼리 개수:", len(eval_qids), "| 코퍼스 크기:", len(index_cids))
    # 계획서 4.3: 검색 지표는 k = 1 / 5 / 10 로 고정.
    # (cfg.top_k=5를 쓰면 k_list=[1,5,5]가 되어 @10이 빠지고 @5가 중복됨)
    result = evaluate(retrieve_fn, eval_qids, gold, k_list=[1, 5, 10], top_k=10)
    for name, value in result.items():
        print(f"{name}: {value:.4f}")
