# ============================================
# Ko-miracl 기반 평가 실행 파일
# - 데이터 로딩(rag.data)과 지표 계산(rag.metrics)은 팀 파이프라인 걸 그대로 재사용
# - 코퍼스는 148만 개 전체를 스트리밍하는 대신, 팀이 미리 뽑아둔
#   data/ko_miracl_subset.jsonl(정답 2,105개 전부 + hard negative 2,895개, 5,000개)을 로컬 로드
# ============================================

import json
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

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "ko_miracl_subset.jsonl"

cfg = GenerationConfig()

queries = load_queries(DATA)
dev_qrels = load_qrels(DATA, DATA.dev_split)

# dev split에서 정답 있는(score>0) 질문 전부 사용 (213개)
n_available = dev_qrels[dev_qrels[DATA.qr_score] > 0][DATA.qr_qid].nunique()
eval_qids = sample_pos_queries(dev_qrels, DATA, n=n_available)
gold = build_gold(dev_qrels, DATA, eval_qids)


def load_local_corpus(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    corpus_text, corpus_title = {}, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus_text[row[DATA.c_id]] = row[DATA.c_text]
            corpus_title[row[DATA.c_id]] = row[DATA.c_title]
    return corpus_text, corpus_title


corpus_text, corpus_title = load_local_corpus(CORPUS_PATH)
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
    result = evaluate(retrieve_fn, eval_qids, gold, k_list=[1, 5, cfg.top_k], top_k=cfg.top_k)
    for name, value in result.items():
        print(f"{name}: {value:.4f}")
