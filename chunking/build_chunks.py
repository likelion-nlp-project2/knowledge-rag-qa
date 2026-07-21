import json
import re
import sys

import ftfy
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SUBSET_PATH = "docs_subset.tsv"
CHUNK_SIZES = [300, 500, 800]  # 단어(word) 기준 chunk size

# ===== 1) passage 기준 정리: doc 단위 → {doc_id, text} =====
def load_docs(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t", 3)
            if len(fields) != 4:
                continue
            doc_id, url, title, body = fields
            docs.append({"doc_id": doc_id, "url": url, "title": title, "body": body})
    return docs


# ===== 2) 텍스트 정제 (불필요 문자 제거) =====
WS_RE = re.compile(r"\s+")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def clean_text(raw: str) -> str:
    text = ftfy.fix_text(raw)                       # 깨진 인코딩(모지바케) 복구: â€™ 등
    if "<" in text and ">" in text:                 # 남아있는 HTML 태그 제거
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = CTRL_RE.sub(" ", text)                    # 제어문자 제거
    text = WS_RE.sub(" ", text).strip()              # 공백/개행 정규화
    return text


def build_passages(docs):
    passages = []
    for d in docs:
        title = clean_text(d["title"])
        body = clean_text(d["body"])
        text = f"{title}. {body}" if title else body
        passages.append({"doc_id": d["doc_id"], "url": d["url"], "text": text})
    return passages


# ===== 3) chunk size 실험 (단어 수 기준, 겹침 없음) =====
def chunk_text(text: str, size: int):
    words = text.split(" ")
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def build_chunks(passages, size):
    documents = []
    next_id = 1
    for p in passages:
        for idx, chunk in enumerate(chunk_text(p["text"], size)):
            if not chunk.strip():
                continue
            documents.append({
                "id": next_id,
                "doc_id": p["doc_id"],
                "chunk_idx": idx,
                "text": chunk,
            })
            next_id += 1
    return documents


def main():
    docs = load_docs(SUBSET_PATH)
    print(f"원본 문서 수: {len(docs)}")

    passages = build_passages(docs)
    print(f"passage 정리 완료: {len(passages)}개")

    for size in CHUNK_SIZES:
        documents = build_chunks(passages, size)
        out_path = f"chunks_{size}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        avg_len = sum(len(d["text"].split()) for d in documents) / len(documents)
        print(f"chunk_size={size}: {len(documents)}개 chunk → {out_path} (평균 {avg_len:.0f} 단어)")


if __name__ == "__main__":
    main()
