# ============================================
# 평가 실행 파일
# 실제로 평가를 실행하고 결과를 출력
# mock_functions import 부분을 추후 진짜 함수 파일로 교체 예정
# ============================================

from test_queries import test_queries
from mock_functions import retrieve, generate_answer  # 진짜 함수로 교체 예정
from evaluate import evaluate_retrieval, evaluate_mrr, evaluate_generation

print("Retrieval hit@5:", evaluate_retrieval(retrieve, test_queries))
print("Retrieval MRR:", evaluate_mrr(retrieve, test_queries))
print("Generation accuracy:", evaluate_generation(retrieve, generate_answer, test_queries))
