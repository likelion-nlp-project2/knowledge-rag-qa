"""위키백과/Wikidata 메타데이터로 서브셋 jsonl을 보강한다 (ENRICH_GUIDE.md 기준 구현).

rag/enrich.py와 동일한 fetch/rollup 로직(fetch_wiki_meta, fetch_wikidata_p31,
fetch_wikidata_labels, rollup_type, _TYPE_RULES)을 그대로 복제해서 쓰되, ChromaDB
collection.update() 대신 각 청크에 메타데이터를 채워서 jsonl로 출력한다.

rag.enrich를 직접 import하지 않는 이유: rag.enrich -> rag.index -> chromadb로
이어지는 import 체인이 있는데, 이 로컬 환경엔 chromadb 빌드에 필요한 Visual
Studio 빌드 도구가 없어서 설치가 안 된다(팀 requirements.txt의 chromadb==0.5.20도
로컬에서 빌드 실패 확인함). 순수 fetch/롤업 로직만 필요하므로 chromadb 의존성 없이
동작하도록 이 파일에 복제해서 사용한다 - 로직은 rag/enrich.py와 동일해야 하며,
그쪽이 바뀌면 이 파일도 맞춰 갱신할 것.

전제 조건(ENRICH_GUIDE.md §0): _id 가 "{위키curid}#{청크번호}" 형식이어야
doc_id 를 위키 pageid 로 바로 조회할 수 있다. 우리 축소 코퍼스는 이미 검증됨
(전량 \\d+#\\d+ 패턴, 예: "985241#0" -> ko.wikipedia.org pageid 985241 = 실제 문서).

출력 필드(원본 _id/title/text 유지 + 추가):
  doc_id, url, doc_type, categories, quality_ok
같은 doc_id의 모든 청크는 doc_id/url/doc_type/categories 값이 동일하고(문서 단위
메타), quality_ok 만 청크 text 길이 기반으로 청크별로 다르다.

배치(pageid 40개)마다 fetch 후 즉시 출력 파일에 append+flush 한다 -> 중간에
중단돼도 --resume(기본값)으로 재실행하면 이미 쓰인 doc_id는 건너뛰고 이어서
진행한다(재fetch 없음).

사용:
  python enrich_corpus.py --in ../data/ko_miracl_reduced_corpus.jsonl \\
      --out ../data/ko_miracl_reduced_corpus_enriched.jsonl
  python enrich_corpus.py --in ... --out ... --limit 100 --dry-run   # 소량 점검(파일 미기록)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag.data import iter_local_corpus  # noqa: E402

_ID_PATTERN = re.compile(r"^\d+#\d+$")

# --- 아래부터 rag/enrich.py와 동일한 fetch/롤업 로직 (chromadb 의존성 없이 복제) ---

WIKI_API = "https://ko.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
UA = "KoMiraclRAG/1.0 (ko-miracl retriever metadata enrichment)"

MIN_QUALITY_CHARS = 30       # 이보다 짧은 청크는 노이즈로 간주(quality_ok=False)
PAGEID_BATCH = 40            # 위키 API 한 번에 조회할 pageid 수 (분류 잘림 방지 위해 <50)
QID_BATCH = 50                # Wikidata 한 번에 조회할 항목 수

SESSION = requests.Session()  # 커넥션 재사용(TLS 핸드셰이크 절약)
_last_call_ts: Dict[str, float] = {}
MIN_INTERVAL = 1.0  # 호스트별 최소 요청 간격(초, --sleep으로 덮어씀). wiki/wikidata는
                     # 서로 다른 서버라 한쪽 호출이 다른쪽 대기시간에 영향 안 주도록 호스트별 추적


def _throttle(host: str) -> None:
    now = time.time()
    wait = MIN_INTERVAL - (now - _last_call_ts.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_call_ts[host] = time.time()

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
    """P31 라벨 목록을 대분류 하나로 축약한다. (rag/enrich.py와 동일)"""
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
        _throttle(url)
        try:
            r = SESSION.get(url, params=params, headers={"User-Agent": UA}, timeout=30)
            if r.status_code == 429:  # rate limit: Retry-After 존중 후 재시도
                wait = max(float(r.headers.get("Retry-After", 3)), 3.0) + 1.5 * attempt
                print(f"  [429] rate limited ({url}), {wait:.0f}초 대기 후 재시도 (attempt {attempt+1})", flush=True)
                time.sleep(min(wait, 60))
                _last_call_ts[url] = time.time()
                last = RuntimeError("429 Too Many Requests")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"요청 실패: {url} ({last})")


def fetch_wiki_meta(doc_ids: List[str]) -> Dict[str, dict]:
    """pageid 배치 -> {doc_id: {categories:[...], qid:Optional[str]}}."""
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
    """Q-id 배치 -> {qid: [P31 target qid,...]}."""
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
    """Q-id 배치 -> {qid: 영문 라벨}."""
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


# --- 여기까지 복제된 rag/enrich.py 로직 ---


def _doc_type_for(w: dict, p31: Dict[str, List[str]], label_cache: Dict[str, str]) -> str:
    """rag/enrich.py의 _doc_type_for와 동일한 규칙(동음이의 우선 -> P31 롤업)."""
    if w.get("is_disambig"):
        return "동음이의"
    qid = w.get("qid")
    labs = [label_cache.get(t, "") for t in p31.get(qid, [])] if qid else []
    return rollup_type(labs)


def check_preconditions(in_path: Path, sample: int = 200) -> None:
    """ENRICH_GUIDE.md §0: _id가 숫자#숫자 형식인지 샘플로 점검."""
    n = 0
    for cid, _title, _text in iter_local_corpus(str(in_path)):
        n += 1
        if not _ID_PATTERN.match(cid):
            raise ValueError(
                f"전제 조건 위반: _id={cid!r} 가 '숫자#숫자' 형식이 아님 "
                "(이 서브셋은 위키백과 pageid 기반이 아닐 수 있음 - ENRICH_GUIDE.md §0 참고)"
            )
        if n >= sample:
            break
    print(f"전제 조건 점검 통과: 샘플 {n}개 전부 '숫자#숫자' 형식")


def _load_done_docs(out_path: Path) -> set:
    """이미 출력 파일에 쓰인 doc_id 집합 (재개용)."""
    done: set = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["doc_id"])
    return done


def run(
    in_path: Path,
    out_path: Path,
    limit_docs: int | None,
    sleep: float,
    dry_run: bool,
    resume: bool,
) -> None:
    global MIN_INTERVAL
    MIN_INTERVAL = sleep  # 호스트별 최소 간격으로 사용 (--sleep 인자 재활용)
    check_preconditions(in_path)

    # 1) 서브셋을 문서(doc_id) 단위로 묶기
    doc_title: Dict[str, str] = {}
    doc_chunks: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for cid, title, text in iter_local_corpus(str(in_path)):
        did = cid.split("#")[0]
        doc_title.setdefault(did, title)
        doc_chunks[did].append((cid, text))

    doc_ids = list(doc_chunks.keys())
    if limit_docs:
        doc_ids = doc_ids[:limit_docs]

    done = _load_done_docs(out_path) if (resume and not dry_run) else set()
    if done:
        print(f"[enrich] resume: 이미 완료 {len(done):,} 문서 건너뜀")
    todo = [d for d in doc_ids if d not in done]
    n_chunks = sum(len(doc_chunks[d]) for d in todo)
    print(
        f"[enrich] 처리 대상 {len(todo):,} 문서 / {n_chunks:,} 청크 "
        f"(전체 {len(doc_ids):,} 문서) {'(dry-run)' if dry_run else ''}"
    )

    # 2) 배치마다: 위키 -> Wikidata P31 -> 라벨 조회 후 즉시 파일에 기록(배치 단위 체크포인트)
    label_cache: Dict[str, str] = {}
    type_dist: Counter = Counter()
    written = 0

    out_mode = "a" if (resume and out_path.exists() and not dry_run) else "w"
    out_f = None if dry_run else open(out_path, out_mode, encoding="utf-8")

    try:
        for i in range(0, len(todo), PAGEID_BATCH):
            batch = todo[i : i + PAGEID_BATCH]

            # 요청 간격은 _get() 안 _throttle()이 호스트별로 알아서 지킨다 (수동 sleep 불필요)
            wiki = fetch_wiki_meta(batch)

            qids = [wiki[d]["qid"] for d in batch if wiki.get(d, {}).get("qid")]
            p31: Dict[str, List[str]] = {}
            for j in range(0, len(qids), QID_BATCH):
                p31.update(fetch_wikidata_p31(qids[j : j + QID_BATCH]))

            targets = sorted({t for ts in p31.values() for t in ts if t not in label_cache})
            for j in range(0, len(targets), QID_BATCH):
                label_cache.update(fetch_wikidata_labels(targets[j : j + QID_BATCH]))
                time.sleep(sleep)

            for did in batch:
                w = wiki.get(did, {})
                dtype = _doc_type_for(w, p31, label_cache)
                type_dist[dtype] += 1
                meta_common = {
                    "doc_id": did,
                    "url": f"https://ko.wikipedia.org/?curid={did}",
                    "doc_type": dtype,
                    "categories": "|".join(w.get("categories", [])),
                }
                for cid, text in doc_chunks[did]:
                    row = {
                        "_id": cid,
                        "title": doc_title.get(did, ""),
                        "text": text,
                        **meta_common,
                        "quality_ok": len(text.strip()) >= MIN_QUALITY_CHARS,
                    }
                    written += 1
                    if out_f:
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if out_f:
                    out_f.flush()  # 문서 단위 체크포인트 -> 중단돼도 이어서 재개 가능

            if (i // PAGEID_BATCH) % 10 == 0:
                done_docs = min(i + PAGEID_BATCH, len(todo))
                print(f"  진행 {done_docs:,}/{len(todo):,} 문서 | 기록 {written:,} 청크", flush=True)
    finally:
        if out_f:
            out_f.close()

    verb = "수집" if dry_run else "기록"
    print(f"[enrich] {'(dry-run) ' if dry_run else ''}완료: {written:,} 청크 {verb} -> {out_path}")
    print(f"[enrich] doc_type 분포(이번 실행): {dict(type_dist)}")


def main() -> None:
    p = argparse.ArgumentParser(description="위키/Wikidata 메타데이터로 서브셋 jsonl 보강 (ENRICH_GUIDE.md)")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--limit", type=int, default=None, help="문서 N개만 처리 (점검용)")
    p.add_argument("--sleep", type=float, default=5.0, help="호스트별 최소 요청 간격(초). 반복 위반 시 429 벌금이 9초->48~51초로 커지는 걸 실측함 - 보수적으로 잡을 것")
    p.add_argument("--dry-run", action="store_true", help="파일 기록 없이 수집만(점검용)")
    p.add_argument(
        "--resume", action="store_true", default=True, help="출력 파일에 이미 있는 문서는 건너뜀(기본값)"
    )
    p.add_argument("--no-resume", dest="resume", action="store_false", help="처음부터 전부 다시 처리")
    a = p.parse_args()
    run(Path(a.in_path), Path(a.out_path), a.limit, a.sleep, a.dry_run, a.resume)


if __name__ == "__main__":
    main()
