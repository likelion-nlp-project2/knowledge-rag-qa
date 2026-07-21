import json
import sys

from build_chunks import load_docs, clean_text, chunk_text

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QRELS_PATH   = "2019qrels-docs_original.txt"
QUERIES_PATH = "msmarco-test2019-queries.tsv"
SUBSET_PATH  = "docs_subset.tsv"
CHUNK_SIZE   = 250
OUT_PATH     = f"chunks_{CHUNK_SIZE}.json"
N_DOCS       = 5000


def load_queries(path):
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, q = line.rstrip("\n").split("\t", 1)
            queries[qid] = q
    return queries


# docs_subset.tsv를 뽑을 때(extract_subset.py)와 동일한 우선순위 로직으로
# docid -> qid(가장 관련도 높았던 쿼리) 매핑을 다시 만든다.
def load_docid_to_qid(path, n_docs):
    scored = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()             # qid Q0 docid rating
            if len(parts) < 4:
                continue
            qid, docid, rating = parts[0], parts[2], int(parts[3])
            scored.append((rating, docid, qid))
    scored.sort(reverse=True)

    docid_to_qid, seen = {}, set()
    for rating, docid, qid in scored:
        if docid not in seen:
            seen.add(docid)
            docid_to_qid[docid] = qid
        if len(seen) >= n_docs:
            break
    return docid_to_qid


def main():
    docid_to_qid = load_docid_to_qid(QRELS_PATH, N_DOCS)
    queries = load_queries(QUERIES_PATH)
    docs = load_docs(SUBSET_PATH)
    print(f"원본 문서 수: {len(docs)}")

    documents = []
    next_id = 1
    skipped = 0
    for d in docs:
        qid = docid_to_qid.get(d["doc_id"])
        if qid is None or qid not in queries:
            skipped += 1
            continue
        title = queries[qid]                 # 실제 검색 쿼리(질문형태)를 title로 사용
        body = clean_text(d["body"])
        for idx, chunk in enumerate(chunk_text(body, CHUNK_SIZE)):
            if not chunk.strip():
                continue
            documents.append({
                "id": next_id,
                "doc_id": d["doc_id"],
                "chunk_idx": idx,
                "title": title,
                "url": d["url"],
                "text": chunk,
            })
            next_id += 1

    if skipped:
        print(f"경고: qid 매칭 안 된 문서 {skipped}개는 제외됨")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    avg_len = sum(len(x["text"].split()) for x in documents) / len(documents)
    print(f"chunk_size={CHUNK_SIZE}: {len(documents)}개 chunk → {OUT_PATH} (평균 {avg_len:.0f} 단어)")


if __name__ == "__main__":
    main()
