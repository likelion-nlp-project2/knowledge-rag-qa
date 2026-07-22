"""Ko-miracl 파이프라인 설정.

다른 데이터셋/모델로 바꿀 때는 이 파일만 수정하면 나머지 모듈은 그대로 재사용 가능하다.
(BEIR 스타일: queries + qrels(score>0=정답) + corpus(id,title,text) 형식 가정)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataSchema:
    dataset: str = "taeminlee/Ko-miracl"
    corpus_config: str = "corpus"
    c_id: str = "_id"
    c_title: str = "title"
    c_text: str = "text"
    queries_config: str = "queries"
    q_id: str = "_id"
    q_text: str = "text"
    qrels_config: str = "default"
    train_split: str = "train"
    dev_split: str = "dev"
    qr_qid: str = "query-id"
    qr_cid: str = "corpus-id"
    qr_score: str = "score"


DATA = DataSchema()
SEED = 42


@dataclass
class FinetuneConfig:
    """노트북1(ko_miracl_finetune_compare): 리트리버 파인튜닝 전/후 비교."""

    base_model: str = "jhgan/ko-sroberta-multitask"
    query_prefix: str = ""      # e5 계열이면 "query: "
    passage_prefix: str = ""    # e5 계열이면 "passage: "
    top_k: int = 10
    k_list: List[int] = field(default_factory=lambda: [1, 5, 10])
    n_train_queries: int = 300
    n_eval_queries: int = 100
    neg_pool_size: int = 3000
    epochs: int = 1
    batch_size: int = 32
    max_seq_len: int = 256
    index_batch: int = 128


@dataclass
class GenerationConfig:
    """노트북2(ko_miracl_rag_generation): 리트리버 + LLM RAG 생성."""

    embed_model: str = "BAAI/bge-m3"
    query_prefix: str = ""
    passage_prefix: str = ""
    llm_name: str = "Qwen/Qwen2.5-7B-Instruct"
    top_k: int = 5
    n_eval_queries: int = 30
    neg_pool_size: int = 3000
    batch_size: int = 128
    max_seq_len: int = 512
