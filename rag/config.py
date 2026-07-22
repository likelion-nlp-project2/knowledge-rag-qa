"""Ko-miracl 파이프라인 설정.

다른 데이터셋/모델로 바꿀 때는 이 파일만 수정하면 나머지 모듈은 그대로 재사용 가능하다.
(BEIR 스타일: queries + qrels(score>0=정답) + corpus(id,title,text) 형식 가정)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


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


# ---------------------------------------------------------------------------
# 검색 임베딩 모델 레지스트리 (교체/비교용)
#
# 새 모델을 비교하고 싶으면 여기 한 줄만 추가하면 된다. ingest/서버는 --model 로
# key 를 골라 쓰고, Chroma 컬렉션은 komiracl_{key} 로 분리 저장되므로 여러 모델을
# 나란히 올려 검색 품질을 비교할 수 있다. (채택 모델: bge-m3)
# ---------------------------------------------------------------------------
@dataclass
class EmbedModelConfig:
    key: str            # 짧은 별칭 (컬렉션 이름/CLI 인자)
    hf_id: str          # HuggingFace / SentenceTransformer 모델 id
    dim: int            # 임베딩 차원 (참고용)
    query_prefix: str = ""      # e5 계열이면 "query: "
    passage_prefix: str = ""    # e5 계열이면 "passage: "
    max_seq_len: int = 512


EMBED_MODELS: Dict[str, EmbedModelConfig] = {
    # 채택 모델
    "bge-m3": EmbedModelConfig("bge-m3", "BAAI/bge-m3", 1024, max_seq_len=512),
    # 비교 후보
    "ko-sroberta": EmbedModelConfig(
        "ko-sroberta", "jhgan/ko-sroberta-multitask", 768, max_seq_len=256
    ),
    "e5-large": EmbedModelConfig(
        "e5-large", "intfloat/multilingual-e5-large", 1024,
        query_prefix="query: ", passage_prefix="passage: ", max_seq_len=512,
    ),
}
DEFAULT_EMBED_MODEL = "bge-m3"


def get_embed_model(key: str) -> EmbedModelConfig:
    if key not in EMBED_MODELS:
        raise KeyError(
            f"unknown embed model {key!r}. 등록된 모델: {list(EMBED_MODELS)} "
            f"(추가하려면 rag/config.py EMBED_MODELS 참고)"
        )
    return EMBED_MODELS[key]


def collection_name(model_key: str) -> str:
    """모델별로 컬렉션을 분리해 비교 가능하게 한다."""
    return f"komiracl_{model_key}"


# ---------------------------------------------------------------------------
# 인프라 설정 (환경변수로 덮어쓸 수 있게 — Docker/Cloudflare 배포용)
# ---------------------------------------------------------------------------
@dataclass
class InfraConfig:
    corpus_path: str = os.getenv(
        "CORPUS_PATH", "data/ko_miracl_reduced_corpus.jsonl"
    )
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
    embed_model: str = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    # RTX 3060(12GB) 기준 기본값. OOM 나면 낮추면 된다.
    ingest_batch: int = int(os.getenv("INGEST_BATCH", "256"))
    fp16: bool = os.getenv("FP16", "1") == "1"


INFRA = InfraConfig()
