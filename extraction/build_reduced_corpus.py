"""BEIR 스타일 축소 코퍼스 생성 (retriever 평가/비교용).

문제의식: qrels에 등장한 corpus-id만 모아서 vectordb에 넣으면
검색 공간이 148만 -> 1.5만으로(약 1%) 줄어들어 "정답이 후보 풀에
반드시 있다"가 보장되는 인공적인 상황이 됨. 실제 RAG 난이도는
무관한 문서 다수를 제치고 정답을 찾는 데서 나오므로 그 부분이
사라지면 리더보드/논문 수치와 비교 불가능.

방식:
1. qrels(train+dev)에 등장한 모든 corpus-id의 원문서(base_id, `#` 앞부분)를
   "필수 포함"으로 지정 -> 판정된 특정 문단만이 아니라 해당 문서의
   모든 문단(형제 청크)을 통째로 포함 (parent 확장 전략 테스트 가능하게)
2. 나머지 예산은 판정 안 된 문서들 중에서 무작위(seed 고정)로 채움
3. 결과: 문서(청크 아님) 단위로 샘플링된 목표 크기(TARGET_TOTAL)의 축소 코퍼스
"""

import json
import random
import sys

import pandas as pd
from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_PATH = "ko_miracl_reduced_corpus.jsonl"
TARGET_TOTAL = 200_000  # 목표 청크(문단) 총 개수
SEED = 42

# 1) qrels 로드 -> 판정된 corpus-id들의 base_id 추출 (score 상관없이 전부: 정답+오답 후보 다 필수)
ds_default = load_dataset("taeminlee/Ko-miracl", "default")
qrels = pd.concat([ds_default["train"].to_pandas(), ds_default["dev"].to_pandas()], ignore_index=True)
judged_base_ids = set(qrels["corpus-id"].str.split("#").str[0].unique())
print(f"qrels에 등장한 고유 문서(base_id): {len(judged_base_ids):,}개")

# 2) corpus 전체를 1차로 훑으며 문서별 청크 수만 집계 (텍스트는 아직 안 모음)
corpus_stream = load_dataset("taeminlee/Ko-miracl", "corpus", split="corpus", streaming=True)
doc_chunk_count: dict[str, int] = {}
for i, row in enumerate(corpus_stream):
    base_id = row["_id"].split("#")[0]
    doc_chunk_count[base_id] = doc_chunk_count.get(base_id, 0) + 1
    if (i + 1) % 300_000 == 0:
        print(f"  1차 스캔 {i+1:,}행 처리")

print(f"corpus 고유 문서 수: {len(doc_chunk_count):,}개")

required_chunks = sum(doc_chunk_count[b] for b in judged_base_ids)
print(f"필수 포함(qrels 판정 문서) 청크 수: {required_chunks:,}개")

# 3) 나머지 예산만큼 판정 안 된 문서 중에서 무작위(seed 고정)로 채움
remaining_pool = [b for b in doc_chunk_count if b not in judged_base_ids]
random.Random(SEED).shuffle(remaining_pool)

selected = set(judged_base_ids)
total = required_chunks
for b in remaining_pool:
    if total >= TARGET_TOTAL:
        break
    selected.add(b)
    total += doc_chunk_count[b]

print(f"최종 선택 문서 수: {len(selected):,}개 -> 청크 합계 {total:,}개 (목표 {TARGET_TOTAL:,})")

# 4) corpus를 2차로 스트리밍하며 선택된 문서의 모든 청크 텍스트를 그대로 기록
count = 0
corpus_stream2 = load_dataset("taeminlee/Ko-miracl", "corpus", split="corpus", streaming=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for i, row in enumerate(corpus_stream2):
        base_id = row["_id"].split("#")[0]
        if base_id in selected:
            f.write(json.dumps({"_id": row["_id"], "title": row["title"], "text": row["text"]}, ensure_ascii=False) + "\n")
            count += 1
        if (i + 1) % 300_000 == 0:
            print(f"  2차 스캔 {i+1:,}행 처리 / {count:,}개 기록")

print(f"완료: {count:,}개 청크 저장 -> {OUT_PATH}")
