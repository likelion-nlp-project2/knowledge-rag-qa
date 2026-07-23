"""로컬 서브셋(ko_miracl_reduced_corpus.jsonl) → 임베딩 → 영속 ChromaDB 적재.

팀원이 만든 20만 청크 서브셋을 스트리밍으로 읽어 배치 임베딩(bge-m3)하고,
영속 Chroma 서버 컬렉션(komiracl_{model})에 upsert 한다.

  # 로컬(데스크탑에서 직접):
  python -m rag.ingest --model bge-m3

  # Docker(권장):
  docker compose run --rm api python -m rag.ingest --model bge-m3

옵션:
  --model     EMBED_MODELS 의 key (기본: 환경변수 EMBED_MODEL 또는 bge-m3)
  --corpus    서브셋 jsonl 경로 (기본: CORPUS_PATH)
  --batch     임베딩/적재 배치 크기 (기본: INGEST_BATCH, RTX3060=256)
  --reset     기존 컬렉션을 지우고 처음부터 다시 적재
  --resume    컬렉션에 이미 있는 id는 건너뛰고 신규만 적재 (중단 후 이어하기)
  --limit     앞에서 N개만 적재 (파이프라인 점검용)

여러 번 실행해도 id 기준 upsert 라 중복 적재되지 않는다(모델을 바꾸면
컬렉션이 분리되므로 나란히 비교 가능).
"""

from __future__ import annotations

import argparse
import time
from typing import List, Tuple

import torch

from .config import INFRA, collection_name, get_embed_model
from .data import count_local_corpus, iter_local_corpus
from .embedding import embed, load_embedder
from .index import connect, get_or_create_collection


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _flush(collection, model, mcfg, buf: List[Tuple[str, str, str]], batch: int) -> int:
    """버퍼에 쌓인 (cid, title, text)를 임베딩해 upsert. 적재 개수 반환."""
    if not buf:
        return 0
    ids = [c for c, _, _ in buf]
    titles = [t for _, t, _ in buf]
    texts = [x for _, _, x in buf]
    embs = embed(model, texts, mcfg.passage_prefix, batch_size=batch)
    collection.upsert(
        ids=ids,
        embeddings=embs.tolist(),
        documents=texts,
        metadatas=[{"title": t} for t in titles],
    )
    return len(ids)


def run(
    model_key: str,
    corpus_path: str,
    batch: int,
    reset: bool,
    limit: int | None,
    resume: bool = False,
) -> None:
    mcfg = get_embed_model(model_key)
    name = collection_name(model_key)
    device = _device()

    print(f"[ingest] model={mcfg.hf_id} device={device} fp16={INFRA.fp16}")
    print(f"[ingest] corpus={corpus_path}")
    print(f"[ingest] chroma={INFRA.chroma_host}:{INFRA.chroma_port} collection={name}")

    client = connect(INFRA.chroma_host, INFRA.chroma_port)
    if reset:
        try:
            client.delete_collection(name)
            print(f"[ingest] reset: '{name}' 삭제됨")
        except Exception:
            pass
    collection = get_or_create_collection(client, name)

    existing: set = set()
    if resume:
        # id 만 가져온다(문서/임베딩 payload 제외해서 가볍게).
        existing = set(collection.get(include=[])["ids"])
        print(f"[ingest] resume: 기존 {len(existing):,}개 id 건너뜀")

    model = load_embedder(mcfg.hf_id, device, mcfg.max_seq_len, fp16=INFRA.fp16)

    total = limit if limit else count_local_corpus(corpus_path)
    target = max(total - len(existing), 1)  # 이번에 새로 적재할 예상치(진행률 기준)
    print(f"[ingest] 대상 청크: {total:,}개 (신규 예상 {target:,}개), 배치: {batch}")

    done = 0
    skipped = 0
    buf: List[Tuple[str, str, str]] = []
    t0 = time.time()
    for i, (cid, title, text) in enumerate(iter_local_corpus(corpus_path)):
        if limit and i >= limit:
            break
        if cid in existing:
            skipped += 1
            continue
        buf.append((cid, title, text))
        if len(buf) >= batch:
            done += _flush(collection, model, mcfg, buf, batch)
            buf = []
            rate = done / max(time.time() - t0, 1e-6)
            eta = (target - done) / max(rate, 1e-6)
            print(
                f"  {done:,}/{target:,} 신규 ({done/target*100:.1f}%) "
                f"| skip {skipped:,} | {rate:.0f} chunk/s | ETA {eta/60:.1f}분",
                flush=True,
            )
    done += _flush(collection, model, mcfg, buf, batch)

    print(f"[ingest] 완료: 신규 {done:,}개 적재, {skipped:,}개 건너뜀, "
          f"컬렉션 총 {collection.count():,}개 ({(time.time()-t0)/60:.1f}분)")


def main() -> None:
    p = argparse.ArgumentParser(description="로컬 서브셋 → ChromaDB 적재")
    p.add_argument("--model", default=INFRA.embed_model, help="EMBED_MODELS key")
    p.add_argument("--corpus", default=INFRA.corpus_path)
    p.add_argument("--batch", type=int, default=INFRA.ingest_batch)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="컬렉션에 이미 있는 id는 건너뛰고 신규만 적재")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    run(args.model, args.corpus, args.batch, args.reset, args.limit, args.resume)


if __name__ == "__main__":
    main()
