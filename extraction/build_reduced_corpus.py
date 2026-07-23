"""BEIR 스타일 축소 코퍼스 생성 (retriever 평가/비교용).

문제의식: qrels에 등장한 corpus-id만 모아서 vectordb에 넣으면
검색 공간이 148만 -> 1.5만으로(약 1%) 줄어들어 "정답이 후보 풀에
반드시 있다"가 보장되는 인공적인 상황이 됨. 실제 RAG 난이도는
무관한 문서 다수를 제치고 정답을 찾는 데서 나오므로 그 부분이
사라지면 리더보드/논문 수치와 비교 불가능.

방식:
1. qrels(train+dev)에 등장한 모든 corpus-id의 원문서(base_id, `#` 앞부분)를
   "필수 포함"으로 지정 -> 판정된 특정 문단만이 아니라 해당 문서의
   모든 문단(형제 청크)을 통째로 포함. 문서 하나가 일부 문단만 들어가면
   나중에 검색된 문단 주변 맥락을 확장해서 보여주는 parent 확장 전략을
   테스트할 수 없기 때문에, 필러 문서도 항상 문서(청크 아님) 단위로 채운다.
2. 나머지 예산은 판정 안 된 문서들 중에서 무작위(seed로 재현 가능)로 채움
3. seed·목표 크기(target)를 인자로 받아, 같은 방식으로 여러 변형을 생성할 수 있다
   (멀티시드 재현성 검증, 5만/10만/20만 단계별 saturation curve 측정 등에 사용)

주의: 판정 문서만으로 이미 필수 포함 청크 수(실행 시 로그로 출력)가 나오므로,
target을 그보다 작게 주면 필러 없이 판정 문서만큼만 만들어진다.

로컬 캐시: corpus 148만 행 전체를 최초 1회만 스트리밍해서
data/corpus_full_cache.jsonl 에 저장해두고, 이후 seed/target을 바꿔 여러 번
돌릴 때는 이 로컬 캐시만 읽는다 (매번 HuggingFace를 다시 스트리밍하지 않음).

사용법:
  python build_reduced_corpus.py                       # seed=42, target=200000 (기본, 팀 공유 파일명 유지)
  python build_reduced_corpus.py --seed 43 --target 200000
  python build_reduced_corpus.py --target 150000
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FULL_CACHE_PATH = DATA_DIR / "corpus_full_cache.jsonl"
DEFAULT_SEED = 42
DEFAULT_TARGET = 200_000


def ensure_full_cache() -> None:
    """corpus 148만 행 전체를 로컬에 한 번만 캐싱. 이미 있으면 재사용."""
    if FULL_CACHE_PATH.exists():
        print(f"로컬 전체 캐시 사용: {FULL_CACHE_PATH}")
        return
    DATA_DIR.mkdir(exist_ok=True)
    print("로컬 캐시 없음 -> corpus 전체 1회 스트리밍 (시간 소요, 이후 재사용됨)")
    stream = load_dataset("taeminlee/Ko-miracl", "corpus", split="corpus", streaming=True)
    with open(FULL_CACHE_PATH, "w", encoding="utf-8") as f:
        for i, row in enumerate(stream):
            f.write(
                json.dumps({"_id": row["_id"], "title": row["title"], "text": row["text"]}, ensure_ascii=False)
                + "\n"
            )
            if (i + 1) % 300_000 == 0:
                print(f"  캐싱 {i+1:,}행 처리")
    print("전체 corpus 로컬 캐싱 완료")


def load_doc_chunk_count() -> dict:
    """base_id -> 청크 수. 로컬 캐시에서 집계 (HF 재접속 없음)."""
    doc_chunk_count: dict[str, int] = {}
    with open(FULL_CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            base_id = json.loads(line)["_id"].split("#")[0]
            doc_chunk_count[base_id] = doc_chunk_count.get(base_id, 0) + 1
    return doc_chunk_count


def build(seed: int, target: int, out_path: Path) -> None:
    ensure_full_cache()

    ds_default = load_dataset("taeminlee/Ko-miracl", "default")
    qrels = pd.concat([ds_default["train"].to_pandas(), ds_default["dev"].to_pandas()], ignore_index=True)
    judged_base_ids = set(qrels["corpus-id"].str.split("#").str[0].unique())
    print(f"qrels에 등장한 고유 문서(base_id): {len(judged_base_ids):,}개")

    doc_chunk_count = load_doc_chunk_count()
    print(f"corpus 고유 문서 수: {len(doc_chunk_count):,}개")

    required_chunks = sum(doc_chunk_count[b] for b in judged_base_ids)
    print(f"필수 포함(qrels 판정 문서) 청크 수: {required_chunks:,}개")
    if target < required_chunks:
        print(
            f"경고: target({target:,})이 필수 포함 청크 수({required_chunks:,})보다 작음 "
            f"-> 필러 없이 판정 문서만으로 {required_chunks:,}개가 생성됨"
        )

    remaining_pool = [b for b in doc_chunk_count if b not in judged_base_ids]
    random.Random(seed).shuffle(remaining_pool)

    selected = set(judged_base_ids)
    total = required_chunks
    for b in remaining_pool:
        if total >= target:
            break
        selected.add(b)
        total += doc_chunk_count[b]

    print(f"최종 선택 문서 수: {len(selected):,}개 -> 청크 합계 {total:,}개 (목표 {target:,}, seed={seed})")

    count = 0
    with open(FULL_CACHE_PATH, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            if row["_id"].split("#")[0] in selected:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

    print(f"완료: {count:,}개 청크 저장 -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BEIR 스타일 Ko-miracl 축소 코퍼스 생성")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--out", type=str, default=None, help="미지정 시 seed/target 기반으로 자동 결정")
    args = parser.parse_args()

    if args.out:
        out_name = args.out
    elif args.seed == DEFAULT_SEED and args.target == DEFAULT_TARGET:
        out_name = "ko_miracl_reduced_corpus.jsonl"  # 팀이 이미 쓰고 있는 기본 파일명 유지
    else:
        out_name = f"ko_miracl_reduced_corpus_seed{args.seed}_{args.target}.jsonl"

    build(args.seed, args.target, DATA_DIR / out_name)


if __name__ == "__main__":
    main()
