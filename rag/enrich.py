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

배치(pageid 40개)마다 fetch 후 '즉시' Chroma 에 기록한다(all-fetch-then-write 가 아님).
그래서 중간에 프로세스가 죽어도 --resume(기본값)으로 재실행하면 이미 doc_id 메타가
붙은 문서는 건너뛰고 이어서 진행한다 — 재fetch/재작업 없음.

사용:
  # 전체 보강 (적재 완료 후 1회, 중단돼도 그대로 재실행하면 이어서 진행)
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
# 규칙은 '우선순위 순'이며(위가 높음), 라벨이 여러 개면 가장 높은 대분류를 택한다.
# 'human' 은 'human settlement' 오분류를 막기 위해 정확히 일치할 때만 인물로 본다(별도 처리).
_TYPE_RULES: List[Tuple[str, List[str]]] = [
    ("목록", ["list article", "wikimedia list"]),
    ("동음이의", ["disambiguation"]),
    ("생물", ["taxon", "species", "breed", "genus", "monotypic"]),
    ("화학물질", ["chemical compound", "chemical element", "chemical", "mineral",
                "protein", "enzyme"]),
    ("장소", [
        "city", "town", "village", "municipality", "county", "province",
        "country", "state", "prefecture", "district", "region", "borough",
        "ward", "dong", "eup", "myeon", "comune", "commune", "township",
        "administrative territorial", "human settlement", "settlement",
        "capital", "metropolitan area", "neighborhood", "international border",
        "mountain", "river", "lake", "island", "peninsula", "bay", "sea",
        "ocean", "canyon", "valley", "desert", "waterfall", "volcano",
        "peak", "hill", "plateau", "cape", "geographic", "location",
        "body of water", "national park", "provincial park", "protected area",
        "building", "skyscraper", "tower", "bridge", "stadium", "arena",
        "hotel", "hospital", "library", "museum", "temple", "church",
        "cathedral", "palace", "castle", "fortress", "shrine", "pagoda",
        "monument", "landmark", "historic", "heritage", "archaeological",
        "square", "park", "airport", "port", "harbor", "station", "halt",
        "interchange", "road", "highway", "route", "street", "railway",
        "subway", "metro", "canal", "dam", "reservoir", "tunnel",
        "post office", "prison", "villa", "cultural heritage", "structure",
        "facility", "campus",
    ]),
    ("사건", [
        "spaceflight", "space mission", "battle", "war", "campaign",
        "event", "election", "tournament", "competition", "olympic",
        "sporting event", "sports season", "season", "incident", "disaster",
        "conflict", "revolution", "treaty", "festival", "ceremony", "summit",
        "holiday", "holy day", "occurrence",
    ]),
    ("작품", [
        "film", "movie", "television series", "television program", "tv series",
        "program", "series", "album", "single", "song", "book", "novel",
        "literary work", "written work", "video game", "painting", "sculpture",
        "artwork", "manga", "anime", "manhwa", "webtoon", "comic", "opera",
        "musical", "play", "poem", "short story", "essay", "magazine",
        "newspaper", "periodical", "journal", "franchise", "website",
        "software", "operating system", "character",
    ]),
    ("조직", [
        "business", "company", "enterprise", "organization", "organisation",
        "nonprofit", "band", "musical group", "team", "political party",
        "university", "school", "college", "agency", "institution",
        "association", "club", "record label", "airline", "military unit",
        "armed forces", "navy", "army", "board of education",
        "government agency", "governmental organization", "bank", "exchange",
    ]),
    ("개념", [
        "concept", "theorem", "doctrine", "distribution", "field of study",
        "academic discipline", "discipline", "academic major",
        "branch of science", "science", "theory", "method", "principle",
        "law", "legal", "penal code", "phenomenon", "process", "unit",
        "profession", "occupation", "position", "title", "genre", "ideology",
        "philosophy", "religion", "language", "writing system", "number",
        "mathematical", "algorithm", "disease", "symptom", "medical", "food",
        "dish", "beverage", "cuisine", "sport", "activity", "era name",
        "period", "era", "bon-gwan", "term", "type", "model", "class",
    ]),
]


def rollup_type(labels: List[str]) -> str:
    """P31 라벨 목록을 대분류 하나로 축약한다.

    - 'human' 정확 일치 → 인물 (최우선)
    - 그 외엔 매칭된 규칙 중 '가장 우선순위 높은(위쪽)' 대분류를 택한다.
    - 아무것도 안 맞으면 '기타'.
    """
    for lab in labels:
        if lab.strip().lower() == "human":
            return "인물"
    best_rank: Optional[int] = None
    best = "기타"
    for lab in labels:
        l = lab.strip().lower()
        if not l:
            continue
        for rank, (name, keys) in enumerate(_TYPE_RULES):
            if any(k in l for k in keys):
                if best_rank is None or rank < best_rank:
                    best_rank, best = rank, name
                break
    return best


def _get(url: str, params: dict, retries: int = 10) -> dict:
    params = {**params, "format": "json", "formatversion": 2}
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params,
                             headers={"User-Agent": UA}, timeout=30)
            if r.status_code == 429:  # rate limit: Retry-After 존중 후 재시도
                wait = max(float(r.headers.get("Retry-After", 3)), 3.0) + 1.5 * attempt
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
        pp = p.get("pageprops", {})
        out[did] = {
            "categories": cats,
            "qid": pp.get("wikibase_item"),
            "is_disambig": "disambiguation" in pp,
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


def _doc_type_for(w: dict, p31: Dict[str, List[str]],
                  label_cache: Dict[str, str]) -> str:
    """한 문서의 wiki 메타 + p31 + 라벨캐시로 doc_type 을 정한다."""
    if w.get("is_disambig"):            # 동음이의: P31 보다 우선(검색 노이즈라 별도 표시)
        return "동음이의"
    qid = w.get("qid")
    labs = [label_cache.get(t, "") for t in p31.get(qid, [])] if qid else []
    return rollup_type(labs)


def _load_done_docs(col) -> set:
    """이미 enrich 된 문서(doc_id 메타데이터가 있는 청크) 집합을 한 번에 스캔한다."""
    done = set()
    got = col.get(include=["metadatas"])
    for cid, m in zip(got.get("ids", []), got.get("metadatas", [])):
        if m and m.get("doc_id"):
            done.add(cid.split("#", 1)[0])
    return done


def run(
    model_key: str,
    corpus_path: str,
    limit_docs: Optional[int],
    update_batch: int,
    sleep: float,
    dry_run: bool,
    resume: bool,
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

    client = connect(INFRA.chroma_host, INFRA.chroma_port)
    col = get_or_create_collection(client, name)

    # 재개: 이미 doc_id 메타데이터가 붙은 문서는 건너뛴다(중간에 죽어도 이어서).
    done: set = set()
    if resume and not dry_run:
        done = _load_done_docs(col)
        print(f"[enrich] resume: 이미 완료 {len(done):,} 문서 건너뜀")
    todo = [d for d in doc_ids if d not in done]
    n_chunks = sum(len(doc_chunks[d]) for d in todo)
    print(f"[enrich] 처리 대상 {len(todo):,} 문서 / {n_chunks:,} 청크 "
          f"(전체 {len(doc_ids):,} 문서) {'(dry-run)' if dry_run else ''}")

    # 2) 배치마다: 위키→Wikidata→라벨 수집 후 '즉시' Chroma 갱신 (배치 단위 체크포인트)
    label_cache: Dict[str, str] = {}   # target Q-id → 영문 라벨 (실행 내 재사용)
    type_dist: Counter = Counter()
    updated = 0
    done_docs = 0

    for i in range(0, len(todo), PAGEID_BATCH):
        batch = todo[i:i + PAGEID_BATCH]

        wiki = fetch_wiki_meta(batch)
        time.sleep(sleep)

        # 이 배치의 Q-id 들에 대해서만 P31 조회
        qids = [wiki[d]["qid"] for d in batch
                if wiki.get(d, {}).get("qid")]
        p31: Dict[str, List[str]] = {}
        for j in range(0, len(qids), QID_BATCH):
            p31.update(fetch_wikidata_p31(qids[j:j + QID_BATCH]))
            time.sleep(sleep)

        # 아직 캐시에 없는 P31 대상 라벨만 조회
        targets = sorted({t for ts in p31.values() for t in ts
                          if t not in label_cache})
        for j in range(0, len(targets), QID_BATCH):
            label_cache.update(fetch_wikidata_labels(targets[j:j + QID_BATCH]))
            time.sleep(sleep)

        # 이 배치 문서들의 청크 metadata 조립
        ids_buf: List[str] = []
        meta_buf: List[dict] = []
        for did in batch:
            w = wiki.get(did, {})
            dtype = _doc_type_for(w, p31, label_cache)
            type_dist[dtype] += 1
            meta_common = {
                "doc_id": did,
                "title": doc_title.get(did, ""),
                "url": f"https://ko.wikipedia.org/?curid={did}",
                "doc_type": dtype,
                "categories": "|".join(w.get("categories", [])),
            }
            for cid, text in doc_chunks[did]:
                ids_buf.append(cid)
                meta_buf.append({
                    "chunk_id": cid,
                    **meta_common,
                    "quality_ok": len(text.strip()) >= MIN_QUALITY_CHARS,
                })

        # 즉시 기록 (update_batch 단위로 쪼개서)
        if ids_buf and not dry_run:
            for k in range(0, len(ids_buf), update_batch):
                col.update(ids=ids_buf[k:k + update_batch],
                           metadatas=meta_buf[k:k + update_batch])
        updated += len(ids_buf)
        done_docs += len(batch)

        if (i // PAGEID_BATCH) % 10 == 0:
            print(f"  진행 {done_docs:,}/{len(todo):,} 문서 | 갱신 {updated:,} 청크",
                  flush=True)

    print(f"[enrich] {'(dry-run) ' if dry_run else ''}완료: {updated:,} 청크 갱신")
    print(f"[enrich] doc_type 분포(이번 실행): {dict(type_dist)}")


def main() -> None:
    p = argparse.ArgumentParser(description="위키/Wikidata 메타데이터로 Chroma 보강")
    p.add_argument("--model", default=INFRA.embed_model, help="EMBED_MODELS key")
    p.add_argument("--corpus", default=INFRA.corpus_path)
    p.add_argument("--limit", type=int, default=None, help="문서 N개만 (점검용)")
    p.add_argument("--batch", type=int, default=500, help="Chroma update 배치 크기")
    p.add_argument("--sleep", type=float, default=0.1, help="API 배치 간 지연(초)")
    p.add_argument("--dry-run", action="store_true", help="DB 갱신 없이 수집만")
    p.add_argument("--resume", action="store_true", default=True,
                   help="이미 doc_id 메타가 붙은 문서는 건너뛰고 이어서 진행(기본값)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="처음부터 전부 다시 enrich")
    a = p.parse_args()
    run(a.model, a.corpus, a.limit, a.batch, a.sleep, a.dry_run, a.resume)


if __name__ == "__main__":
    main()
