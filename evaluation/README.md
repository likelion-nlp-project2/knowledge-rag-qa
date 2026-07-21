# 평가 기준 (Evaluation Criteria)

chunk_size(250 / 300 / 500 / 800) 비교 시 "어떤 게 더 좋은지"를 판단하는 규칙.
**결과 숫자를 보기 전에** 미리 확정해 둠 (사후에 유리한 지표를 갖다 붙이는 편향 방지).

## 지표 정의

| 지표 | 의미 | 방향 |
|---|---|---|
| **hit@k** | 정답 문서(chunk)가 검색 상위 k개 안에 있으면 1, 없으면 0 → 전체 평균 | 높을수록 좋음 |
| **MRR** | 정답이 몇 등에 나왔는지 반영 (1등=1.0, 2등=0.5, 3등=0.33 … 없으면 0) | 높을수록 좋음 |

- 정답 판정: **doc_id 매칭**(retrieve 결과의 `metadata.doc_id` == gold `docid`)을 우선.
  retrieve가 doc_id를 안 주면 **gold_answer 텍스트 부분일치**로 fallback (대소문자 무시).
- 구현: [evaluate.py](evaluate.py)의 `evaluate_retrieval`, `evaluate_mrr`.

## 메인 지표

**`hit@k`를 메인으로 사용.** 단, 이 **`k`는 RAG가 LLM에 실제로 넣는 chunk 개수와 동일하게** 맞춘다.

- 이유: chunk_size 비교의 목적은 "정답이 LLM context 안에 들어오게 하는 것".
  RAG가 top-k를 통째로 넣으므로, "정답 chunk가 그 안에 들어왔나"를 재는 hit@k가 결과와 가장 직결된다.
- 현재 기본값: **hit@5** (RAG가 top-5를 넣는다고 가정. 실제 k가 바뀌면 이 값도 맞출 것).
- MRR을 메인으로 두지 않는 이유: RAG는 top-k를 통째로 넣어 1등이든 3등이든 다 context에 들어가므로,
  1차 판단은 hit@k가 더 적합. MRR은 아래 동점 처리용 보조 지표로 사용.

## 동점 처리 규칙

메인 지표(hit@k)가 같을 때 다음 순서로 우열을 가린다:

1. **hit@k** — 높은 쪽 승 (메인)
2. **동점이면 → MRR** — 높은 쪽 승
   (정답을 더 상위에 올리는 쪽이 k를 줄여도 안전하므로)
3. **그래도 동점이면 → chunk_size 작은 쪽** — 승
   (저장 용량·검색 속도·LLM 토큰 비용 유리, context도 더 정밀)

→ 이 3단계로 항상 하나의 chunk_size가 유일하게 결정된다.

## 참고: 아직 미정 / 대기 중

- **생성(RAG 답변) 평가 방식**(자동 vs 사람 준정량 채점): 멘토링 후 결정 예정.
  현재 `evaluate_generation`은 자동(gold 문구 포함 여부)으로 임시 동작.
- **chunk_size 4개 비교 실행부**: 진짜 retrieve 함수 + chunk별 Chroma 인덱스가 준비되면
  `run_eval.py`를 chunk_size 루프로 감싸 위 기준대로 표를 뽑는다.
