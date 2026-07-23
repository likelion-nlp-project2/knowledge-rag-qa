"""Ko-miracl 파이프라인 실행 진입점.

  python -m rag.cli compare          # 리트리버 파인튜닝 전/후 비교
  python -m rag.cli ask "질문"        # 질의재작성 -> 검색 -> 생성 RAG
"""

from __future__ import annotations

import argparse
import random
from typing import Optional

import chromadb
import numpy as np
import pandas as pd
import torch

from .config import DATA, SEED, FinetuneConfig, GenerationConfig
from .data import (
    build_gold,
    collect_corpus,
    load_qrels,
    load_queries,
    needed_corpus_ids,
    sample_pos_queries,
)
from .embedding import load_embedder
from .finetune import build_training_examples, finetune
from .generation import rag_answer
from .index import build_collection, retrieve
from .metrics import evaluate


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_compare(cfg: FinetuneConfig = FinetuneConfig(), seed: int = SEED) -> pd.DataFrame:
    """파인튜닝 전/후 리트리버 성능을 같은 평가셋으로 비교한다."""
    random.seed(seed)
    np.random.seed(seed)
    device = _device()

    queries = load_queries(DATA)
    train_qrels = load_qrels(DATA, DATA.train_split)
    dev_qrels = load_qrels(DATA, DATA.dev_split)

    train_qids = sample_pos_queries(train_qrels, DATA, cfg.n_train_queries, seed)
    eval_qids = sample_pos_queries(dev_qrels, DATA, cfg.n_eval_queries, seed)
    train_gold = build_gold(train_qrels, DATA, train_qids)
    gold = build_gold(dev_qrels, DATA, eval_qids)

    needed = needed_corpus_ids(train_gold, gold)
    corpus_text, _ = collect_corpus(DATA, needed, cfg.neg_pool_size)
    index_cids = list(corpus_text.keys())

    model = load_embedder(cfg.base_model, device, cfg.max_seq_len)
    client = chromadb.Client()

    def make_retrieve_fn(collection):
        def _retrieve(qid, k):
            hits = retrieve(collection, model, queries[qid], cfg.query_prefix, k=k)
            return [h["corpus_id"] for h in hits]

        return _retrieve

    col_before = build_collection(
        client, "retr-before", model, index_cids, corpus_text,
        passage_prefix=cfg.passage_prefix, batch_size=cfg.index_batch,
    )
    before = evaluate(make_retrieve_fn(col_before), eval_qids, gold, cfg.k_list, cfg.top_k)

    examples = build_training_examples(
        train_qids, train_gold, queries, corpus_text, cfg.query_prefix, cfg.passage_prefix
    )
    finetune(model, examples, cfg.epochs, cfg.batch_size)

    col_after = build_collection(
        client, "retr-after", model, index_cids, corpus_text,
        passage_prefix=cfg.passage_prefix, batch_size=cfg.index_batch,
    )
    after = evaluate(make_retrieve_fn(col_after), eval_qids, gold, cfg.k_list, cfg.top_k)

    comp = pd.DataFrame({"before": before, "after": after})
    comp["Δ(after-before)"] = comp["after"] - comp["before"]
    comp["개선%"] = np.where(
        comp["before"] > 0, comp["Δ(after-before)"] / comp["before"] * 100, np.nan
    )
    return comp


def run_ask(
    question: str,
    cfg: GenerationConfig = GenerationConfig(),
    seed: int = SEED,
    k: Optional[int] = None,
    modes: tuple = ("strict",),
) -> dict:
    """Ko-miracl dev 코퍼스 부분집합을 색인하고, 질문 하나에 대해 RAG 응답을 생성한다.

    modes에 여러 개를 주면 색인/모델은 한 번만 올리고 프롬프트만 바꿔 나란히 답한다.
    """
    random.seed(seed)
    np.random.seed(seed)
    device = _device()

    dev_qrels = load_qrels(DATA, DATA.dev_split)
    eval_qids = sample_pos_queries(dev_qrels, DATA, cfg.n_eval_queries, seed)
    gold = build_gold(dev_qrels, DATA, eval_qids)

    needed = needed_corpus_ids(gold)
    corpus_text, corpus_title = collect_corpus(DATA, needed, cfg.neg_pool_size)
    index_cids = list(corpus_text.keys())

    model = load_embedder(cfg.embed_model, device, cfg.max_seq_len)
    client = chromadb.Client()
    collection = build_collection(
        client, "ko-miracl-rag", model, index_cids, corpus_text, corpus_title,
        passage_prefix=cfg.passage_prefix, batch_size=cfg.batch_size,
    )

    result = None
    for mode in modes:
        result = rag_answer(
            question, collection, model, cfg.query_prefix, k or cfg.top_k, mode
        )
        if mode == modes[0]:
            print("① 원 질문     :", result["question"])
            print("① 재작성 질의 :", result["rewritten"])
            print("② 검색 결과 (score = 1 - distance):")
            for i, c in enumerate(result["contexts"], 1):
                print(f"   [{i}] score={c['score']:.4f} id={c['corpus_id']} | {c['title']}")
        print(f"\n④ 생성 응답 [{mode}] :\n" + result["answer"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ko-miracl RAG 파이프라인")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("compare", help="리트리버 파인튜닝 전/후 비교")

    p_ask = sub.add_parser("ask", help="질의재작성 -> 검색 -> 생성 RAG")
    p_ask.add_argument("question")
    p_ask.add_argument("--k", type=int, default=None)

    args = parser.parse_args()
    if args.cmd == "compare":
        comp = run_compare()
        print(comp.round(4))
    elif args.cmd == "ask":
        run_ask(args.question, k=args.k)


if __name__ == "__main__":
    main()
