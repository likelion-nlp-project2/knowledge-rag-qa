# ============================================
# 선택한 docid들의 전체 body 텍스트 확인
# - manual_queries 만들기 전에 정확한 내용 파악용
# ============================================

import pandas as pd
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

docs_path = os.path.join(data_dir, "docs_subset.tsv")
df = pd.read_csv(docs_path, sep="\t", header=None,
                  names=["docid", "url", "title", "body"])

# 질문 만들기 좋아 보이는 docid 골라둔 목록
selected_docids = [
    "D499866",  # grey vs white matter
    "D804064",  # spruce tree
]

for docid in selected_docids:
    row = df[df["docid"] == docid]
    if not row.empty:
        print("docid:", docid)
        print("Title:", row["title"].values[0])
        print("Full Body:", row["body"].values[0])
        print("=" * 80)