# ============================================
# 가짜(mock) retrieve / generate 함수
# - retrieval, RAG 생성 함수가 아직 완성되지 않아서
#   임시로 만들어둔 대체 함수
# - 평가 파이프라인(evaluate.py, run_eval.py)이 에러 없이 작동하는지만 확인하는 용도
# - 실제 팀원 함수 완성되면 이 파일은 삭제하고, run_eval.py에서 import 경로 교체
# ============================================

def retrieve(query):
    return [{"text": "에펠탑은 1889년 프랑스 만국박람회를 위해 지어졌다."}]

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

def generate_answer(query, context):
    return "1889년에 지어졌습니다."