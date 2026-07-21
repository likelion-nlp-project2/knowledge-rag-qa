# ============================================
# 가짜(mock) retrieve / generate 함수
# - retrieval, RAG 생성 함수가 아직 완성되지 않아서
#   임시로 만들어둔 대체 함수
# - 평가 파이프라인(evaluate.py, run_eval.py)이 에러 없이 작동하는지만 확인하는 용도
# - 실제 팀원 함수 완성되면 이 파일은 삭제하고, run_eval.py에서 import 경로 교체
# ============================================

def retrieve(query):
    # 진짜 retrieve도 이 형식으로 리턴하면 됨: text, score, doc_id
    return [{"text": "에펠탑은 1889년 프랑스 만국박람회를 위해 지어졌다.", "score": 0.8, "doc_id": "D000000"}]

def generate_answer(query, context):
    return "1889년에 지어졌습니다."