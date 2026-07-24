"""[Colab에서 실행] 여러 corpus jsonl 파일의 문서(doc_id) 합집합에 대해
위키백과/Wikidata 메타데이터를 딱 한 번만 조회해서 doc_metadata.jsonl로 저장한다.

독립 실행 스크립트(rag 패키지 의존성 없음, requests만 있으면 됨) - Colab에
repo를 클론하지 않고 이 파일 + 업로드한 jsonl들만으로 바로 돌아간다.

왜 파일별로 따로 안 돌리나: seed42의 125k/150k/175k는 20만(seed42)의
부분집합이고(같은 seed라 필러 문서까지 겹침), seed43/44도 정답 문서 7,692개는
동일하게 공유한다. 파일마다 따로 enrich하면 같은 문서를 최대 3~4번 중복
조회하게 되므로, 모든 입력 파일의 doc_id 합집합에 대해 한 번만 모은다.
(125k/150k/175k는 입력에 안 넣어도 됨 - 20만 결과에 이미 다 포함됨)

배치(pageid 40개)마다 즉시 append+flush -> 중단돼도 재실행하면 이미 쓰인
doc_id는 건너뛰고 이어서 진행(resume, 기본 동작).

Colab 사용법:
  1) 왼쪽 파일 탭에 아래 3개 jsonl 업로드 (또는 구글드라이브 마운트 후 경로 수정):
     - ko_miracl_reduced_corpus.jsonl              (seed42 @ 20만, 메인)
     - ko_miracl_reduced_corpus_seed43_200000.jsonl
     - ko_miracl_reduced_corpus_seed44_200000.jsonl
  2) 이 셀 그대로 실행 (런타임 끊겨도 doc_metadata.jsonl은 세션 디스크에 남아있고,
     재실행하면 이어서 진행됨 - 단, 런타임 자체가 초기화되면 파일도 날아가니
     중간중간 doc_metadata.jsonl을 다운로드해서 백업해둘 것)
  3) 완료되면 doc_metadata.jsonl을 로컬로 다운로드 -> apply_metadata.py로
     7개 파일 전부에 조인(로컬, API 호출 없음, 몇 초~몇십 초)
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass  # Colab stdout은 이미 UTF-8 (reconfigure 불필요)

# ── 설정 (필요시 여기만 수정) ────────────────────────────────────────────
INPUT_FILES = [
    "ko_miracl_reduced_corpus.jsonl",
    "ko_miracl_reduced_corpus_seed43_200000.jsonl",
    "ko_miracl_reduced_corpus_seed44_200000.jsonl",
]
OUT_PATH = "doc_metadata.jsonl"
# 위키/Wikidata 속도제한이 IP 단위라, 로컬(even)/Colab(odd)에서 동시에 돌리면
# 서로 다른 IP라 독립적으로 처리량이 나옴(대략 2배). 혼자 다 돌릴 땐 "all".
#   "even": pageid(doc_id)가 짝수인 문서만 처리
#   "odd" : pageid(doc_id)가 홀수인 문서만 처리
#   "all" : 분할 없이 전체 처리
SHARD = "even"
# 호스트별 최소 요청 간격(초). 처음엔 9초였던 429 벌금이 반복 위반으로 48~51초까지
# 늘어난 걸 실측함 - 서버가 위반 이력을 누적 추적하며 페널티를 키우는 것으로 보임.
# 1.0초도 부족했으므로 훨씬 보수적으로 잡음 (쿨다운 후 재시작 시 사용).
MIN_INTERVAL = 5.0
# ─────────────────────────────────────────────────────────────────────

WIKI_API = "https://ko.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
UA = "KoMiraclRAG/1.0 (ko-miracl retriever metadata enrichment)"
PAGEID_BATCH = 40
QID_BATCH = 50
SESSION = requests.Session()  # 커넥션 재사용(TLS 핸드셰이크 절약)
_last_call_ts: Dict[str, float] = {}


def _throttle(host: str) -> None:
    """호스트별로 최소 MIN_INTERVAL초 간격을 보장 (다른 호스트끼리는 서로 안 기다림)."""
    now = time.time()
    wait = MIN_INTERVAL - (now - _last_call_ts.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_call_ts[host] = time.time()

# rag/enrich.py 와 동일한 doc_type 롤업 규칙(팀 기준 구현 그대로 복제)
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
            if r.status_code == 429:
                wait = max(float(r.headers.get("Retry-After", 3)), 3.0) + 1.5 * attempt
                print(f"  [429] rate limited ({url}), {wait:.0f}초 대기 후 재시도 (attempt {attempt+1})", flush=True)
                time.sleep(min(wait, 60))
                _last_call_ts[url] = time.time()  # 429 자체도 요청이었으므로 다음 호출도 간격 유지
                last = RuntimeError("429 Too Many Requests")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"요청 실패: {url} ({last})")


def fetch_wiki_meta(doc_ids: List[str]) -> Dict[str, dict]:
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
    out: Dict[str, List[str]] = {}
    data = _get(WIKIDATA_API, {"action": "wbgetentities", "ids": "|".join(qids), "props": "claims"})
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
    if not qids:
        return {}
    data = _get(WIKIDATA_API, {
        "action": "wbgetentities", "ids": "|".join(qids), "props": "labels", "languages": "en",
    })
    out: Dict[str, str] = {}
    for qid, ent in data.get("entities", {}).items():
        out[qid] = ent.get("labels", {}).get("en", {}).get("value", "")
    return out


def iter_doc_ids(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield row["_id"].split("#")[0]


def load_done(out_path: str) -> set:
    done: set = set()
    if Path(out_path).exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["doc_id"])
    return done


def main() -> None:
    doc_ids_set: set = set()
    for path in INPUT_FILES:
        before = len(doc_ids_set)
        for did in iter_doc_ids(path):
            doc_ids_set.add(did)
        print(f"{path}: 누적 고유 문서 {len(doc_ids_set):,}개 (+{len(doc_ids_set) - before:,})")
    doc_ids = list(doc_ids_set)
    print(f"입력 {len(INPUT_FILES)}개 파일 합집합: 고유 문서 {len(doc_ids):,}개")

    if SHARD == "even":
        doc_ids = [d for d in doc_ids if int(d) % 2 == 0]
    elif SHARD == "odd":
        doc_ids = [d for d in doc_ids if int(d) % 2 == 1]
    print(f"SHARD={SHARD!r} 적용 후 담당 문서: {len(doc_ids):,}개")

    done = load_done(OUT_PATH)
    todo = [d for d in doc_ids if d not in done]
    print(f"이미 완료 {len(done):,}개 건너뜀 -> 처리 대상 {len(todo):,}개")

    label_cache: Dict[str, str] = {}
    out_f = open(OUT_PATH, "a", encoding="utf-8")
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

            for did in batch:
                w = wiki.get(did, {})
                if w.get("is_disambig"):
                    dtype = "동음이의"
                else:
                    qid = w.get("qid")
                    labs = [label_cache.get(t, "") for t in p31.get(qid, [])] if qid else []
                    dtype = rollup_type(labs)
                row = {
                    "doc_id": did,
                    "url": f"https://ko.wikipedia.org/?curid={did}",
                    "doc_type": dtype,
                    "categories": "|".join(w.get("categories", [])),
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i // PAGEID_BATCH) % 10 == 0:
                print(f"  진행 {min(i + PAGEID_BATCH, len(todo)):,}/{len(todo):,}", flush=True)
    finally:
        out_f.close()

    print(f"완료 -> {OUT_PATH} (총 {len(done) + len(todo):,} 문서)")


if __name__ == "__main__":
    main()
