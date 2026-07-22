"""Ko-MIRACL corpus에서 5,000개 서브셋 추출.

- score=1(관련) 문서는 손대지 않고 전부 포함 (2,105개, qrels에 존재하는 전부).
- score=0(비관련) 문서는 "hard negative" 기준으로 순위를 매겨 채운다:
  qrels는 (query-id, corpus-id) 쌍을 판정한 것이므로, 각 비관련 문서를
  원래 짝지어진 쿼리 텍스트와 문자 2-gram Jaccard 유사도로 비교해
  "쿼리와 헷갈릴 만큼 비슷한데 비관련으로 판정된" 문서를 우선 채택한다.
  (임의 순서/정렬 알고리즘 부산물이 아니라, 검색 평가에서 표준적으로
  쓰이는 hard negative mining 방식)
"""

import sys

import pandas as pd
from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_PATH = "ko_miracl_subset.jsonl"
N_DOCS = 5000


def char_bigrams(text: str) -> set:
    text = text.replace(" ", "")
    return {text[i:i + 2] for i in range(len(text) - 1)} or {text}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# 1) qrels 로드, score=1 / score=0 분리
ds_default = load_dataset("taeminlee/Ko-miracl", "default")
qrels = pd.concat([ds_default["train"].to_pandas(), ds_default["dev"].to_pandas()], ignore_index=True)

pos_cids = set(qrels[qrels["score"] > 0]["corpus-id"].unique())
neg_rows = qrels[qrels["score"] == 0][["query-id", "corpus-id"]]
neg_cids = set(neg_rows["corpus-id"].unique()) - pos_cids  # 같은 문서가 다른 쿼리에선 관련(score=1)일 수 있음 -> 겹치면 pos 우선, neg에서 제외
neg_rows = neg_rows[neg_rows["corpus-id"].isin(neg_cids)]

print(f"score=1(관련) 고유 문서: {len(pos_cids)}개 (전부 포함, 순위 안 매김)")
print(f"score=0(비관련) 후보: {len(neg_cids)}개 -> hard negative 유사도로 순위 매겨서 상위 {N_DOCS - len(pos_cids)}개 선택")

# 2) query 텍스트 로드 (score=0 후보들과 유사도 비교용)
queries_df = load_dataset("taeminlee/Ko-miracl", "queries", split="queries").to_pandas()
query_text = dict(zip(queries_df["_id"], queries_df["text"]))
query_bigrams = {qid: char_bigrams(t) for qid, t in query_text.items()}

# corpus-id -> 짝지어진 query-id 목록 (여러 쿼리에 걸쳐 비관련 판정됐을 수 있음)
neg_cid_to_qids: dict[str, list[str]] = {}
for _, row in neg_rows.iterrows():
    neg_cid_to_qids.setdefault(row["corpus-id"], []).append(row["query-id"])

# 3) corpus를 한 번 스트리밍하며 필요한 텍스트(양성+음성 후보) 전부 수집
needed_cids = pos_cids | neg_cids
corpus_stream = load_dataset("taeminlee/Ko-miracl", "corpus", split="corpus", streaming=True)

collected: dict[str, dict] = {}
for i, row in enumerate(corpus_stream):
    if row["_id"] in needed_cids:
        collected[row["_id"]] = {"_id": row["_id"], "title": row["title"], "text": row["text"]}
        if len(collected) >= len(needed_cids):
            break
    if (i + 1) % 300_000 == 0:
        print(f"  corpus 스캔 {i+1:,}행 / {len(collected)}개 확보")

print(f"corpus 텍스트 확보 완료: {len(collected)}/{len(needed_cids)}")

# 4) score=0 후보에 대해 hard negative 유사도 점수 계산 (해당 corpus-id와 짝지어진 쿼리들 중 최댓값)
neg_scored = []
for cid in neg_cids:
    if cid not in collected:
        continue
    cid_bigrams = char_bigrams(collected[cid]["text"][:300])  # 앞부분만 사용 (효율)
    best = max(
        (jaccard(cid_bigrams, query_bigrams[qid]) for qid in neg_cid_to_qids[cid] if qid in query_bigrams),
        default=0.0,
    )
    neg_scored.append((best, cid))

neg_scored.sort(key=lambda x: x[0], reverse=True)
n_neg_needed = N_DOCS - len(pos_cids)
top_neg_cids = [cid for _, cid in neg_scored[:n_neg_needed]]

print(f"hard negative 상위 {len(top_neg_cids)}개 선정 (유사도 최고 {neg_scored[0][0]:.3f} ~ 최저 {neg_scored[n_neg_needed-1][0]:.3f})")

# 5) 최종 조합: score=1 전부 + hard negative 상위 N개
final_ids = list(pos_cids) + top_neg_cids
rows = [collected[cid] for cid in final_ids if cid in collected]

print(f"최종 추출: {len(rows)}개")

out = pd.DataFrame(rows)
out.to_json(OUT_PATH, orient="records", lines=True, force_ascii=False)
print("저장:", OUT_PATH)
