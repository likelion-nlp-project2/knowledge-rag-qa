
# 평가 로직


def _get_doc_id(doc):
    """
    retrieve 결과 한 건에서 doc_id를 꺼냄.
    - 형식이 팀마다 조금씩 다를 수 있어 여러 위치를 다 확인:
      맨 위 레벨(doc_id / docid) 또는 중첩된 metadata / metadatas 안(Chroma 기본 형식)
    - 못 찾으면 None → 텍스트 매칭으로 fallback
    """
    for key in ("doc_id", "docid"):
        if doc.get(key) is not None:
            return doc[key]
    for meta_key in ("metadata", "metadatas"):
        meta = doc.get(meta_key)
        if isinstance(meta, dict):
            for key in ("doc_id", "docid"):
                if meta.get(key) is not None:
                    return meta[key]
    return None


def _is_hit(tq, doc):
    """
    정답 판정 (한 문서가 정답인지)
    - 1순위: doc_id 매칭 — retrieve 결과에서 doc_id를 찾을 수 있고 gold에도 docid가 있으면
      "같은 문서에서 나왔나"로 정확히 판정 (MS-MARCO 표준, 오차 없음)
    - fallback: 텍스트 부분일치 — doc_id를 못 찾으면 gold 문구가
      본문에 들어있는지로 판정 (대소문자 무시)
    """
    doc_id = _get_doc_id(doc)
    if doc_id is not None and tq.get("docid") is not None:
        return doc_id == tq["docid"]
    return tq["gold_answer"].lower() in doc["text"].lower()


def evaluate_retrieval(retrieve_fn, test_queries, k=5):
    """
    Retrieval 평가 (hit@k): 정답이 top-k 문서 안에 포함되어 있는지 확인
    반환값: hit@k 비율 (0~1 사이, 높을수록 좋음)
    """
    results = []
    for tq in test_queries:
        docs = retrieve_fn(tq["query"])
        hit = any(_is_hit(tq, d) for d in docs[:k])
        results.append(hit)
    return sum(results) / len(results)


def evaluate_mrr(retrieve_fn, test_queries):
    """
    Retrieval 평가 (MRR): 정답 문서가 몇 번째 순위에 나왔는지까지 반영
    - 1등이면 1.0, 2등이면 0.5, 3등이면 0.33 ... 없으면 0
    반환값: MRR 점수 (0~1 사이, 높을수록 좋음)
    """
    scores = []
    for tq in test_queries:
        docs = retrieve_fn(tq["query"])
        rank = None
        for i, d in enumerate(docs):
            if _is_hit(tq, d):
                rank = i + 1  # 순위는 1부터 시작
                break
        scores.append(1 / rank if rank else 0)
    return sum(scores) / len(scores)


def evaluate_generation(retrieve_fn, generate_fn, test_queries):
    """
    생성 평가: retrieve한 문서를 바탕으로 만든 답변에 정답이 포함되어 있는지 확인
    - 생성 평가는 "답변 텍스트"를 보는 거라 doc_id 매칭이 아니라
      gold_answer 텍스트 포함 여부로 판정 (대소문자 무시)
    반환값: 정확도 (0~1 사이, 높을수록 좋음)
    """
    scores = []
    for tq in test_queries:
        docs = retrieve_fn(tq["query"])
        context = [d["text"] for d in docs]
        answer = generate_fn(tq["query"], context)
        score = tq["gold_answer"].lower() in answer.lower()
        scores.append(score)
    return sum(scores) / len(scores)
