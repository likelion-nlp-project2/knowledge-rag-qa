"""RAG 응답 대량 생성 (실험 1단계) — GPT-4o-mini + 검색 API.

Ko-miracl 정답 있는 쿼리(train+dev ≈ 1,000건)를 데모 앱과 동일한 파이프라인
(rag_engine.run_pipeline: 질의재작성 → 검색 API → 스레숄드 → 프롬프트 → 생성)으로
돌려 result/generations_{prompt}.jsonl 에 저장한다.

- 1줄 1건 append + qid 체크포인트: 중단돼도 재실행하면 이어서 돈다.
- 비용(개략): 1,000건 × (입력 ~3k + 출력 ~0.3k 토큰) ≈ $1 미만 (GPT-4o-mini).
- 사전 조건: .env 에 OPENAI_API_KEY, SEARCH_API_URL (검색 API가 떠 있어야 함).

  python evaluation/generate_answers.py --limit 5        # 스모크
  python evaluation/generate_answers.py                  # 전체 (fewshot)
  python evaluation/generate_answers.py --prompt basic   # 프롬프트 A/B용
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.config import DATA, SEED
from rag.data import load_qrels, load_queries, sample_pos_queries
from rag.llm import LLM_MODEL
from rag_engine import run_pipeline
from schema import PipelineConfig

RESULT_DIR = Path(__file__).resolve().parent.parent / "result"


def collect_items() -> tuple[dict, list[tuple[str, str]]]:
    """(queries, [(split, qid)]) — score>0 쿼리 전부, seed 고정 순서."""
    queries = load_queries(DATA)
    items = []
    for split in (DATA.train_split, DATA.dev_split):
        qrels = load_qrels(DATA, split)
        for qid in sample_pos_queries(qrels, DATA, n=10**9, seed=SEED):
            items.append((split, qid))
    return queries, items


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 응답 대량 생성 (GPT-4o-mini)")
    ap.add_argument("--prompt", choices=("basic", "fewshot"), default="fewshot")
    ap.add_argument("--k-context", type=int, default=3,
                    help="LLM에 넣을 문서 수 — 컨텍스트가 비용의 대부분이라 작게 유지")
    ap.add_argument("--k-retrieval", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.55)
    ap.add_argument("--limit", type=int, default=None, help="스모크용: 앞 N건만")
    args = ap.parse_args()

    cfg = PipelineConfig(
        top_k_retrieval=args.k_retrieval, top_k_context=args.k_context,
        min_score=args.min_score, prompt_style=args.prompt,
    )
    out = RESULT_DIR / f"generations_{args.prompt}.jsonl"
    out.parent.mkdir(exist_ok=True)

    done: set = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["qid"] for line in f if line.strip()}

    queries, items = collect_items()
    if args.limit:
        items = items[: args.limit]
    todo = [(s, q) for s, q in items if q not in done]
    print(f"대상 {len(items)}건 / 완료 {len(items) - len(todo)}건 / 남음 {len(todo)}건 → {out}")
    print(f"모델 {LLM_MODEL} · 프롬프트 {args.prompt} · k={args.k_context}")

    with open(out, "a", encoding="utf-8") as f:
        for i, (split, qid) in enumerate(todo, 1):
            resp = run_pipeline(queries[qid], cfg)
            rec = {
                "qid": qid,
                "split": split,
                "question": resp.query,
                "rewritten": resp.rewritten_query,
                "answer": resp.generated_answer,
                "contexts": [c.text for c in resp.filtered_chunks],
                "context_doc_ids": [c.doc_id for c in resp.filtered_chunks],
                "prompt_style": args.prompt,
                "model": LLM_MODEL,
                "config": {"k_context": args.k_context, "k_retrieval": args.k_retrieval,
                           "min_score": args.min_score},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()   # 체크포인트: 중단돼도 완료분은 보존
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} (qid={qid})")


if __name__ == "__main__":
    main()
