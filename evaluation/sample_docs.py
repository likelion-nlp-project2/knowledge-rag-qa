# ============================================
# 5000개 문서 중 질문 만들기 좋은 문서 샘플링
# 공식 qrels로 매칭 안 된 나머지 질문을 
# 직접 만들기 위해 문서 내용을 훑어보는 용도
# ============================================

import pandas as pd
import os
import random

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

docs_path = os.path.join(data_dir, "docs_subset.tsv")
df = pd.read_csv(docs_path, sep="\t", header=None,
                  names=["docid", "url", "title", "body"])

# 무작위로 30개 문서 뽑아서 미리보기
random.seed(60)  # 매번 같은 결과 나오게 고정 (재현 가능하게)
sample_indices = random.sample(range(len(df)), 30)

for idx in sample_indices:
    print("docid:", df.iloc[idx]["docid"])
    print("Title:", df.iloc[idx]["title"])
    print("Body:", df.iloc[idx]["body"][:200])
    print("---")