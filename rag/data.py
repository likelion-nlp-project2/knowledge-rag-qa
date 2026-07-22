"""Ko-miracl 로드 + 학습/평가용 부분집합 구성.

전체 코퍼스(149만 청크)를 다 올리지 않고, 정답 문서 + 네거티브 풀만 스트리밍으로 모은다.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd
from datasets import load_dataset

from .config import DataSchema


def load_queries(schema: DataSchema) -> Dict[str, str]:
    dd = load_dataset(schema.dataset, schema.queries_config)
    split = list(dd.keys())[0]
    return {r[schema.q_id]: r[schema.q_text] for r in dd[split]}


def load_qrels(schema: DataSchema, split: str) -> pd.DataFrame:
    dd = load_dataset(schema.dataset, schema.qrels_config)
    return pd.DataFrame(dd[split])


def sample_pos_queries(
    qrels_df: pd.DataFrame, schema: DataSchema, n: int, seed: int = 42
) -> List[str]:
    """score>0인 쿼리 중 n개를 무작위로 뽑는다."""
    pos = qrels_df[qrels_df[schema.qr_score] > 0]
    qids = pos[schema.qr_qid].unique().tolist()
    random.Random(seed).shuffle(qids)
    return qids[:n]


def build_gold(
    qrels_df: pd.DataFrame, schema: DataSchema, qids: Iterable[str]
) -> Dict[str, Dict[str, float]]:
    """qid -> {cid: score} 매핑 (score>0=정답)."""
    gold: Dict[str, Dict[str, float]] = {}
    subset = qrels_df[qrels_df[schema.qr_qid].isin(qids)]
    for _, row in subset.iterrows():
        gold.setdefault(row[schema.qr_qid], {})[row[schema.qr_cid]] = float(row[schema.qr_score])
    return gold


def needed_corpus_ids(*golds: Dict[str, Dict[str, float]]) -> Set[str]:
    """정답 문서(score>0) id 집합만 모은다 (여러 gold를 합쳐서)."""
    return {
        cid
        for gold in golds
        for rels in gold.values()
        for cid, score in rels.items()
        if score > 0
    }


def collect_corpus(
    schema: DataSchema, needed_cids: Set[str], neg_pool_size: int
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """코퍼스를 스트리밍하며 정답 문서 + 네거티브 풀만 수집한다.

    반환: (cid -> text, cid -> title)
    """
    corpus_dd = load_dataset(schema.dataset, schema.corpus_config, streaming=True)
    stream = corpus_dd[list(corpus_dd.keys())[0]]

    text: Dict[str, str] = {}
    title: Dict[str, str] = {}
    found: Set[str] = set()
    neg = 0
    for row in stream:
        cid = row[schema.c_id]
        if cid in needed_cids and cid not in text:
            text[cid] = row[schema.c_text]
            title[cid] = row[schema.c_title]
            found.add(cid)
        elif neg < neg_pool_size:
            text[cid] = row[schema.c_text]
            title[cid] = row[schema.c_title]
            neg += 1
        if len(found) >= len(needed_cids) and neg >= neg_pool_size:
            break
    return text, title
