# ============================================
# 평가셋 주제 분포 점검 — 213개 질문이 여러 주제에 퍼져 있나? (멘토 피드백 #11)
#
# 주제 라벨은 팀 벡터DB(rag-chroma)의 doc_type을 그대로 쓴다.
#   우리가 만든 분류가 아니라 DB에 이미 있는 값이라 자의성 논란이 없다.
#
# 방식: 질문의 "정답 문서"(qrels 기준) 제목으로 DB를 검색해, doc_id가 일치하는
#   문서의 doc_type을 읽는다. 질문을 검색해 나온 문서를 쓰면 검색 성능이 라벨을
#   오염시키므로(순환) 반드시 정답 문서 기준으로 붙인다.
#
# 한계: 이 API는 doc_id 직접 조회가 없어 제목 검색으로 우회하고, 결과를 3~4개만
#   돌려주므로 문서 단위 매칭률이 100%가 아니다(100건 표본에서 64%).
#   다만 질문당 정답 문서가 평균 2.36개라 "하나만 걸리면" 되므로 질문 단위
#   커버리지는 그보다 높다. 못 찾은 질문은 '미확인'으로 남긴다
#   (= doc_type이 없다는 뜻이 아니라, 우리가 확인하지 못했다는 뜻).
#
# 실행:
#   python evaluation/check_evalset_topics.py
# ============================================

import collections
import json
import math
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

from rag.config import DATA, SEED
from rag.data import build_gold, load_qrels, load_queries, sample_pos_queries

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "ko_miracl_reduced_corpus.jsonl"
OUT_CSV = Path(__file__).resolve().parent / "data" / "evalset_topics.csv"

SEARCH_API = "https://rag-chroma.howcansea.com/search"
API_SLEEP = 0.1
API_TIMEOUT = 60
UNKNOWN = "미확인"


def load_pool():
    queries = load_queries(DATA)
    dev = load_qrels(DATA, DATA.dev_split)
    n_pos = dev[dev[DATA.qr_score] > 0][DATA.qr_qid].nunique()
    qids = sample_pos_queries(dev, DATA, n=n_pos, seed=SEED)   # 다른 평가 스크립트와 동일 순서
    gold = build_gold(dev, DATA, qids)
    gold_cids = {q: [c for c, s in gold[q].items() if s > 0] for q in qids}
    return queries, qids, gold_cids


def load_titles(gold_cids):
    want = {c for cs in gold_cids.values() for c in cs}
    titles = {}
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row[DATA.c_id] in want:
                titles[row[DATA.c_id]] = row[DATA.c_title]
    return titles


def fetch_doc_type(title: str, doc_id: str) -> str | None:
    """제목으로 검색해 doc_id가 일치하는 문서의 doc_type을 돌려준다. 못 찾으면 None."""
    try:
        r = requests.get(SEARCH_API, params={"q": title, "rerank": "none"}, timeout=API_TIMEOUT)
        if r.status_code != 200:
            return None
        for hit in r.json().get("results", []):
            m = hit.get("metadata", {})
            if str(m.get("doc_id")) == str(doc_id):
                return (m.get("doc_type") or "").strip() or None
    except requests.RequestException:
        return None
    return None


def evenness(counts) -> float:
    """0~1. 1에 가까울수록 주제가 고르게 퍼져 있음."""
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    ps = [c / total for c in counts if c > 0]
    return -sum(p * math.log(p) for p in ps) / math.log(len(counts))


def main():
    queries, qids, gold_cids = load_pool()
    titles = load_titles(gold_cids)
    n_docs = sum(len(v) for v in gold_cids.values())
    print(f"질문 {len(qids)}개 / 정답 문서 {n_docs}건 조회 시작 (약 {n_docs*0.6/60:.0f}~{n_docs*2/60:.0f}분)\n")

    rows, done = [], 0
    for qid in qids:
        found = []
        for cid in gold_cids[qid]:
            dt = fetch_doc_type(titles.get(cid, ""), cid.split("#")[0])
            done += 1
            if dt:
                found.append(dt)
            time.sleep(API_SLEEP)
        # 질문 하나에 정답 문서가 여러 개면 가장 많이 나온 유형을 대표로
        topic = collections.Counter(found).most_common(1)[0][0] if found else UNKNOWN
        rows.append({"qid": qid, "question": queries[qid], "topic": topic,
                     "n_gold": len(gold_cids[qid]), "n_matched": len(found)})
        if len(rows) % 25 == 0:
            cov = sum(r["topic"] != UNKNOWN for r in rows) / len(rows) * 100
            print(f"  질문 {len(rows)}/{len(qids)} (문서 {done}/{n_docs}) 커버리지 {cov:.0f}%")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ---- 리포트 (길이 분포와 같은 형태) ----
    vc = df["topic"].value_counts()
    share = (vc / len(df) * 100).round(1)
    print(f"\n===== 평가셋 주제 분포 (질문 {len(df)}개, doc_type 기준) =====")
    for t, n in vc.items():
        print(f"  {t:<8} {n:>4}개  {share[t]:>5.1f}%")

    known = df[df["topic"] != UNKNOWN]
    print(f"\n문서 단위 매칭률 : {df['n_matched'].sum()}/{df['n_gold'].sum()}"
          f" ({df['n_matched'].sum()/df['n_gold'].sum()*100:.1f}%)")
    print(f"질문 단위 커버리지: {len(known)}/{len(df)} ({len(known)/len(df)*100:.1f}%)")

    if len(known):
        kvc = known["topic"].value_counts()
        kshare = (kvc / len(known) * 100)
        print(f"\n--- 확인된 {len(known)}개 기준 ---")
        print(f"주제 종류      : {len(kvc)}종")
        print(f"상위 3개 집중도: {kshare.head(3).sum():.1f}%")
        print(f"고른 정도      : {evenness(kvc.tolist()):.3f}  (1에 가까울수록 고르게 분포)")
        if kshare.head(3).sum() >= 80 or evenness(kvc.tolist()) < 0.6:
            print("판정: 특정 주제에 쏠림 있음 -> 결과 해석 시 명시 필요")
        else:
            print("판정: 여러 주제에 비교적 고르게 분포")
    print(f"\n저장: {OUT_CSV}")


if __name__ == "__main__":
    main()
