# ============================================
# 실제 MS-MARCO 데이터로 test_queries 만들기
# - 데이터 담당이 준 5000개 문서 안에서
#   정답을 찾을 수 있는 질문만 필터링
# - 결과를 test_queries.py에 붙여넣어 사용
# ============================================

import pandas as pd
import os

# 이 스크립트 파일이 있는 폴더(evaluation/) 기준으로 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

# 1. 데이터 담당한테 받은 5000개 문서
docs_path = os.path.join(data_dir, "docs_subset.tsv")
df = pd.read_csv(docs_path, sep="\t", header=None,
                  names=["docid", "url", "title", "body"])

# 2. 질문 목록 (MS-MARCO 공식 dev set)
queries_path = os.path.join(data_dir, "queries.docdev.tsv")
queries = pd.read_csv(queries_path, sep="\t", header=None,
                       names=["qid", "query"])

# 3. 정답 매칭 정보 (qid - docid 매칭, qrels 포맷은 공백 구분)
qrels_path = os.path.join(data_dir, "msmarco-docdev-qrels.tsv")
qrels = pd.read_csv(qrels_path, sep=" ", header=None,
                     names=["qid", "unused", "docid", "relevance"])

print("문서 개수:", df.shape)
print("질문 개수:", queries.shape)
print("정답매칭 개수:", qrels.shape)

# 4. 내가 가진 5000개 문서 안에 정답이 있는 질문만 필터링
my_docids = set(df["docid"].tolist())
valid_qrels = qrels[qrels["docid"].isin(my_docids)]
print("매칭되는 질문 개수:", len(valid_qrels))

# 5. 질문 텍스트 붙이기
merged = valid_qrels.merge(queries, on="qid")
print(merged.head())

# 6. 결과를 CSV로 저장해서 나중에 test_queries.py 만들 때 참고
output_path = os.path.join(data_dir, "matched_queries.csv")
merged.to_csv(output_path, index=False, encoding="utf-8-sig")
print("저장 완료:", output_path)