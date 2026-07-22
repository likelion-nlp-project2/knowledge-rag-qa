"""위키백과/Wikidata 메타데이터로 Chroma 컬렉션을 보강한다 (재임베딩 없이 metadata만 갱신).

서브셋의 각 문서(doc_id = corpus_id 의 '#' 앞부분)에 대해 한국어 위키백과 API +
Wikidata API 에서 구조화 메타데이터를 모아, 그 문서에 속한 모든 청크의 Chroma metadata 를
아래 필드로 덮어쓴다 (임베딩 벡터는 건드리지 않는다).

  chunk_id, doc_id, title, url, doc_type, categories, quality_ok

- doc_type   : Wikidata instance-of(P31) 를 대분류로 롤업
               (인물/장소/작품/사건/조직/생물/화학물질/기타)
- categories : 한국어 위키 분류(hidden 제외), '|' 로 join (Chroma metadata 는 스칼라만 허용)
- url        : https://ko.wikipedia.org/?curid={doc_id}
- quality_ok : 청크 text 길이 기반 노이즈 필터 (표 잔여물 등 배제)

사용:
  # 전체 보강 (적재 완료 후 1회)
  docker compose run --rm api python -m rag.enrich --model bge-m3
  # 소량 점검 (수집만, DB 미변경)
  docker compose run --rm api python -m rag.enrich --model bge-m3 --limit 100 --dry-run
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import requests

from .config import INFRA, collection_name
from .data import iter_local_corpus
from .index import connect, get_or_create_collection

WIKI_API = "https://ko.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
UA = "KoMiraclRAG/1.0 (ko-miracl retriever metadata enrichment)"

MIN_QUALITY_CHARS = 30       # 이보다 짧은 청크는 노이즈로 간주(quality_ok=False)
PAGEID_BATCH = 40            # 위키 API 한 번에 조회할 pageid 수 (분류 잘림 방지 위해 <50)
QID_BATCH = 50              # Wikidata 한 번에 조회할 항목 수

# Wikidata instance-of(P31) 영문 라벨 → 대분류 롤업. 부분일치(소문자) 규칙.
# 'human' 은 'human settlement' 오분류를 막기 위해 정확히 일치할 때만 인물로 본다.
_TYPE_RULES: List[Tuple[str, List[str]]] = [
    ("생물", ["taxon", "species", "breed", "genus"]),
    ("화학물질", ["chemical compound", "chemical element", "chemical"]),
    ("장소", ["city", "town", "village", "municipality", "county", "province",
              "country", "state", "mountain", "river", "lake", "island",
              "administrative territorial", "human settlement", "region",
              "geographic", "location", "capital", "district", "prefecture"]),
    ("작품", ["film", "television series", "tv series", "album", "single",
              "song", "book", "literary work", "novel", "video game",
              "painting", "manga", "anime", "written work", "musical",
              "artwork", "franchise", "play", "comic"]),
    ("사건", ["spaceflight", "space mission", "battle", "war", "event",
              "election", "tournament", "competition", "olympic",
              "sporting event", "incident", "disaster", "conflict"]),
    ("조직", ["business", "company", "enterprise", "organization",
              "organisation", "nonprofit", "band", "sports team",
              "political party", "university", "school", "agency",
              "institution", "association", "club"]),
]


def _match_label(label: str) -> Optional[str]:
    l = label.strip().lower()
    if not l:
        return None
    if l == "human":
        return "인물"
    for name, keys in _TYPE_RULES:
        if any(k in l for k in keys):
            return name
    return None


def rollup_type(labels: List[str]) -> str:
    """P31 라벨 목록을 대분류 하나로 축약한다. 아무것도 안 맞으면 '기타'."""
    for lab in labels:
        m = _match_label(lab)
        if m:
            return m
    return "기타"


def _get(url: str, params: dict, retries: int = 5) -> dict:
    params = {**params, "format": "json", "formatversion": 2}
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params,
                             headers={"User-Agent": UA}, timeout=30)
            if r.status_code == 429:  # rate limit: Retry-After 존중 후 재시도
                wait = float(r.headers.get("Retry-After", 5)) + 2.0 * attempt
                time.sleep(min(wait, 60))
                last = RuntimeError("429 Too Many Requests")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"요청 실패: {url} ({last})")


def fetch_wiki_meta(doc_ids: List[str]) -> Dict[str, dict]:
    """pageid 배치 → {doc_id: {categories:[...], qid:Optional[str]}}."""
    out: Dict[str, dict] = {}
    data = _get(WIKI_API, {
        "action": "query",
        "pageids": "|".join(doc_ids),
        "prop": "categories|pageprops",
        "cllimit": "max",
        "clshow": "!hidden",
    })
    for p in data.get("query", {}).get("pages", []):
        if p.get("missing"):
            continue
        did = str(p["pageid"])
        cats = [c["title"].split(":", 1)[-1] for c in p.get("categories", [])]
        out[did] = {
            "categories": cats,
            "qid": p.get("pageprops", {}).get("wikibase_item"),
        }
    return out


def fetch_wikidata_p31(qids: List[str]) -> Dict[str, List[str]]:
    """Q-id 배치 → {qid: [P31 target qid,...]}."""
    out: Dict[str, List[str]] = {}
    data = _get(WIKIDATA_API, {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "claims",
    })
    for qid, ent in data.get("entities", {}).items():
        targets: List[str] = []
        for claim in ent.get("claims", {}).get("P31", []):
            try:
                targets.append(claim["mainsnak"]["datavalue"]["value"]["id"])
            except (KeyError, TypeError):
                continue
        out[qid] = targets
    return out


def fetch_wikidata_labels(qids: List[str]) -> Dict[str, str]:
    """Q-id 배치 → {qid: 영문 라벨}."""
    if not qids:
        return {}
    data = _get(WIKIDATA_API, {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "labels",
        "languages": "en",
    })
    out: Dict[str, str] = {}
    for qid, ent in data.get("entities", {}).items():
        out[qid] = ent.get("labels", {}).get("en", {}).get("value", "")
    return out


def run(
    model_key: str,
    corpus_path: str,
    limit_docs: Optional[int],
    update_batch: int,
    sleep: float,
    dry_run: bool,
) -> None:
    name = collection_name(model_key)

    # 1) 코퍼스를 문서 단위로 묶기 (청크는 DB 에 실제 적재된 것과 동일: 빈 text 제외)
    doc_title: Dict[str, str] = {}
    doc_chunks: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for cid, title, text in iter_local_corpus(corpus_path):
        did = cid.split("#")[0]
        doc_title.setdefault(did, title)
        doc_chunks[did].append((cid, text))

    doc_ids = list(doc_chunks.keys())
    if limit_docs:
        doc_ids = doc_ids[:limit_docs]
    n_chunks = sum(len(doc_chunks[d]) for d in doc_ids)
    print(f"[enrich] 대상 문서 {len(doc_ids):,}개 / 청크 {n_chunks:,}개 "
          f"{'(dry-run)' if dry_run else ''}")

    # 2) 위키 메타(분류 + Wikidata Q-id) 수집
    wiki: Dict[str, dict] = {}
    for i in range(0, len(doc_ids), PAGEID_BATCH):
        wiki.update(fetch_wiki_meta(doc_ids[i:i + PAGEID_BATCH]))
        if i // PAGEID_BATCH % 20 == 0:
            print(f"  [wiki] {min(i + PAGEID_BATCH, len(doc_ids)):,}/{len(doc_ids):,}",
                  flush=True)
        time.sleep(sleep)

    # 3) Wikidata: P31 대상 수집 → 라벨 해석 → doc_type 롤업
    qids = [m["qid"] for m in wiki.values() if m.get("qid")]
    p31: Dict[str, List[str]] = {}
    for i in range(0, len(qids), QID_BATCH):
        p31.update(fetch_wikidata_p31(qids[i:i + QID_BATCH]))
        time.sleep(sleep)
    target_ids = sorted({t for ts in p31.values() for t in ts})
    labels: Dict[str, str] = {}
    for i in range(0, len(target_ids), QID_BATCH):
        labels.update(fetch_wikidata_labels(target_ids[i:i + QID_BATCH]))
        time.sleep(sleep)

    doc_type: Dict[str, str] = {}
    for did in doc_ids:
        qid = wiki.get(did, {}).get("qid")
        labs = [labels.get(t, "") for t in p31.get(qid, [])] if qid else []
        doc_type[did] = rollup_type(labs)

    # 4) 청크별 metadata 조립 후 Chroma 갱신 (임베딩은 건드리지 않음)
    client = connect(INFRA.chroma_host, INFRA.chroma_port)
    col = get_or_create_collection(client, name)

    ids_buf: List[str] = []
    meta_buf: List[dict] = []
    updated = 0
    type_dist: Counter = Counter()

    def _flush():
        nonlocal updated, ids_buf, meta_buf
        if not ids_buf:
            return
        if not dry_run:
            col.update(ids=ids_buf, metadatas=meta_buf)
        updated += len(ids_buf)
        ids_buf, meta_buf = [], []

    for did in doc_ids:
        w = wiki.get(did, {})
        meta_common = {
            "doc_id": did,
            "title": doc_title.get(did, ""),
            "url": f"https://ko.wikipedia.org/?curid={did}",
            "doc_type": doc_type.get(did, "기타"),
            "categories": "|".join(w.get("categories", [])),
        }
        type_dist[meta_common["doc_type"]] += 1
        for cid, text in doc_chunks[did]:
            ids_buf.append(cid)
            meta_buf.append({
                "chunk_id": cid,
                **meta_common,
                "quality_ok": len(text.strip()) >= MIN_QUALITY_CHARS,
            })
            if len(ids_buf) >= update_batch:
                _flush()
                print(f"  [update] {updated:,}/{n_chunks:,} 청크", flush=True)
    _flush()

    print(f"[enrich] {'(dry-run) ' if dry_run else ''}완료: {updated:,}개 청크 metadata 갱신")
    print(f"[enrich] doc_type 분포: {dict(type_dist)}")


def main() -> None:
    p = argparse.ArgumentParser(description="위키/Wikidata 메타데이터로 Chroma 보강")
    p.add_argument("--model", default=INFRA.embed_model, help="EMBED_MODELS key")
    p.add_argument("--corpus", default=INFRA.corpus_path)
    p.add_argument("--limit", type=int, default=None, help="문서 N개만 (점검용)")
    p.add_argument("--batch", type=int, default=500, help="Chroma update 배치 크기")
    p.add_argument("--sleep", type=float, default=0.1, help="API 배치 간 지연(초)")
    p.add_argument("--dry-run", action="store_true", help="DB 갱신 없이 수집만")
    a = p.parse_args()
    run(a.model, a.corpus, a.limit, a.batch, a.sleep, a.dry_run)


if __name__ == "__main__":
    main()
