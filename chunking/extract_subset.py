import gzip
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # cp949 콘솔에서 UnicodeEncodeError 방지

# ===== 경로 설정 (파일이 같은 폴더에 있으면 이대로) =====
QRELS_PATH   = "2019qrels-docs_original.txt"
DOCS_GZ_PATH = "msmarco-docs.tsv.gz"
OUT_PATH     = "docs_subset.tsv"
N_DOCS       = 5000

# 1) qrels에서 docid 모으기 (관련도 높은 문서 우선)
scored = []
with open(QRELS_PATH, encoding="utf-8") as f:
    for line in f:
        parts = line.split()             # 형식: qid Q0 docid rating
        if len(parts) < 4:
            continue
        rating, docid = int(parts[3]), parts[2]
        scored.append((rating, docid))
scored.sort(reverse=True)

target, seen = [], set()
for rating, docid in scored:
    if docid not in seen:
        seen.add(docid); target.append(docid)
    if len(target) >= N_DOCS:
        break
target_set = set(target)
print(f"목표 문서 수: {len(target_set)}")

# 2) .gz 스트리밍 추출 (다 찾으면 조기 종료)
found = 0
t0 = time.time()
with gzip.open(DOCS_GZ_PATH, "rt", encoding="utf-8", errors="replace") as fin, \
     open(OUT_PATH, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin, 1):
        docid = line.split("\t", 1)[0]
        if docid in target_set:
            fout.write(line); found += 1
            if found == len(target_set):
                break
        if i % 500_000 == 0:
            elapsed = time.time() - t0
            print(f"  {i:,}줄 처리 / {found}개 발견 / {elapsed:.1f}초 경과")

print(f"추출 완료: {found}개 → {OUT_PATH} ({time.time()-t0:.1f}초 소요)")
if found < len(target_set):
    print(f"경고: 목표 {len(target_set)}개 중 {len(target_set) - found}개는 gz 파일에서 찾지 못함")

# 3) 결과 확인
with open(OUT_PATH, encoding="utf-8") as f:
    first_line = f.readline().rstrip("\n")
    fields = first_line.split("\t", 3)
    if len(fields) == 4:
        docid, url, title, body = fields
        print("docid:", docid)
        print("title:", title)
        print("body :", body[:200], "...")
    else:
        print(f"경고: 첫 줄 필드 수가 예상과 다름 ({len(fields)}개)")
        print(first_line[:300])
