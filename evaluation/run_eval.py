# ============================================
# 평가 실행 파일
# 실제로 평가를 실행하고 결과를 출력
# mock_functions import 부분을 추후 진짜 함수 파일로 교체 예정
# ============================================

from test_queries import test_queries_retrieval, test_queries_generation
from mock_functions import retrieve, generate_answer  # 진짜 함수로 교체 예정
from evaluate import evaluate_retrieval, evaluate_mrr, evaluate_generation

# retrieval 지표는 30개 전체, generation(사람 채점)은 앞 15개만 사용
print("Retrieval hit@5:", evaluate_retrieval(retrieve, test_queries_retrieval))
print("Retrieval MRR:", evaluate_mrr(retrieve, test_queries_retrieval))
print("Generation accuracy:", evaluate_generation(retrieve, generate_answer, test_queries_generation))
