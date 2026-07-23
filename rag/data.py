"""Ko-miracl 로드 + 학습/평가용 부분집합 구성.

- queries/qrels: HF(taeminlee/Ko-miracl)에서 로드 (평가 정답셋)
- corpus: 팀원이 만든 로컬 서브셋(ko_miracl_reduced_corpus.jsonl, 20만 청크)을
  스트리밍으로 읽는다. (더 이상 전체 코퍼스를 HF에서 스트리밍하지 않는다)
"""

from __future__ import annotations

import json
import random
from typing import Dict, Iterable, Iterator, List, Set, Tuple

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


# ---------------------------------------------------------------------------
# 로컬 서브셋(jsonl) 로더 — ChromaDB 적재용
# 형식: {"_id": "985055#0", "title": "...", "text": "..."} 한 줄당 한 청크
# ---------------------------------------------------------------------------
def iter_local_corpus(
    path: str, schema: DataSchema = DataSchema()
) -> Iterator[Tuple[str, str, str]]:
    """서브셋 jsonl을 한 줄씩 스트리밍한다. yield (cid, title, text).

    20만 청크를 메모리에 다 올리지 않기 위해 제너레이터로 흘려보낸다.
    빈 text(공백 포함)는 임베딩 노이즈라 건너뛴다.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get(schema.c_text) or "").strip()
            if not text:
                continue
            cid = row[schema.c_id]
            title = row.get(schema.c_title) or ""
            yield cid, title, text


def count_local_corpus(path: str, schema: DataSchema = DataSchema()) -> int:
    """진행률 표시용 총 청크 수(빈 text 제외)."""
    return sum(1 for _ in iter_local_corpus(path, schema))


def load_local_corpus(
    path: str, schema: DataSchema = DataSchema()
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """서브셋 전체를 메모리에 올린다 (compare/평가에서 collect_corpus 대체용).

    반환: (cid -> text, cid -> title)
    """
    text: Dict[str, str] = {}
    title: Dict[str, str] = {}
    for cid, t, x in iter_local_corpus(path, schema):
        text[cid] = x
        title[cid] = t
    return text, title
