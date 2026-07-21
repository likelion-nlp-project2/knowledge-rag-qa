
# 평가 로직 

def evaluate_retrieval(retrieve_fn, test_queries, k=5):
    """
    Retrieval 평가 (hit@k): 정답이 top-k 문서 안에 포함되어 있는지 확인
    반환값: hit@k 비율 (0~1 사이)
    """
    results = []
    for tq in test_queries:
        docs = retrieve_fn(tq["query"])
        hit = any(tq["gold_answer"] in d["text"] for d in docs[:k])
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
            if tq["gold_answer"] in d["text"]:
                rank = i + 1  # 순위는 1부터 시작
                break
        scores.append(1 / rank if rank else 0)
    return sum(scores) / len(scores)


def evaluate_generation(retrieve_fn, generate_fn, test_queries):
    """
    생성 평가: retrieve한 문서를 바탕으로 만든 답변에 정답이 포함되어 있는지 확인
    반환값: 정확도 (0~1 사이)
    """
    scores = []
    for tq in test_queries:
        docs = retrieve_fn(tq["query"])
        context = [d["text"] for d in docs]
        answer = generate_fn(tq["query"], context)
        score = tq["gold_answer"] in answer
        scores.append(score)
    return sum(scores) / len(scores)