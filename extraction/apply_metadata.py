"""doc_metadata.jsonl(위키/Wikidata 메타, API 호출 없음)을 corpus jsonl에 조인해서
enriched jsonl을 만든다. 순수 로컬 조인이라 파일 하나당 몇 초~몇십 초면 끝난다.

colab_fetch_wiki_metadata.py로 문서(doc_id) 단위 메타데이터를 한 번 모아두면,
그걸 여러 corpus 변형(seed/크기 다른 파일들)에 이 스크립트로 반복 적용할 수 있다
(같은 문서를 여러 번 API 조회하지 않기 위함 - ENRICH_GUIDE.md 취지 + 우리 상황:
seed42의 12.5만/15만/17.5만은 20만의 부분집합, seed43/44도 정답 문서 7,692개를 공유).

사용:
  python apply_metadata.py --meta ../data/doc_metadata.jsonl \
      --in ../data/ko_miracl_reduced_corpus.jsonl \
      --out ../data/ko_miracl_reduced_corpus_enriched.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MIN_QUALITY_CHARS = 30


def load_meta(path: Path) -> Dict[str, dict]:
    meta: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                meta[row["doc_id"]] = row
    return meta


def apply(in_path: Path, out_path: Path, meta: Dict[str, dict]) -> None:
    missing: set = set()
    n = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            did = row["_id"].split("#")[0]
            m = meta.get(did)
            if m is None:
                missing.add(did)
                continue
            out = {
                "_id": row["_id"],
                "title": row["title"],
                "text": row["text"],
                "doc_id": did,
                "url": m["url"],
                "doc_type": m["doc_type"],
                "categories": m["categories"],
                "quality_ok": len(row["text"].strip()) >= MIN_QUALITY_CHARS,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"{in_path.name} -> {out_path.name}: {n:,} 청크 기록")
    if missing:
        print(f"  경고: 메타데이터 없는 doc_id {len(missing):,}건 (해당 청크는 건너뜀 - meta 파일이 이 corpus의 문서를 다 못 커버함)")


def main() -> None:
    p = argparse.ArgumentParser(description="doc_metadata.jsonl을 corpus jsonl에 로컬 조인")
    p.add_argument("--meta", required=True, help="colab_fetch_wiki_metadata.py 결과 파일")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    a = p.parse_args()
    meta = load_meta(Path(a.meta))
    print(f"메타데이터 로드: {len(meta):,} 문서")
    apply(Path(a.in_path), Path(a.out_path), meta)


if __name__ == "__main__":
    main()
