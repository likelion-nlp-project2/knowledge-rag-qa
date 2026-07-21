# ============================================
# 평가용 테스트 쿼리셋
# - 공식 MS-MARCO qrels 매칭분(14개) + 직접 만든 질문(16개) = 총 30개
# - gold_answer는 전부 "정답 chunk 안에 들어있을 텍스트 문구"로 통일
#   (retrieve가 id 없이 text만 주므로 → evaluate.py에서 텍스트 부분일치 채점)
# - docid는 원본 MS-MARCO 문서 id (참고/디버깅용, 채점엔 안 씀)
# - 모든 gold_answer는 해당 문서 본문에 존재함을 검증 완료
# ============================================

# 1. 공식 qrels 매칭분 (gold_answer = 문서 본문에서 뽑은 정답 문구)
#    ※ 일부(spruce 목록형, bsrn 뉘앙스, nurse=NP 기준)는 질문-문서 정합이 약함
official_queries = [
    {"query": "cost per person for ptsd treatment", "gold_answer": "2,310 per person", "docid": "D3518488", "source": "official"},
    {"query": "difference between a spruce and a fir tree", "gold_answer": "cones on a fir tree stand straight up", "docid": "D1114554", "source": "official"},
    {"query": "different kinds of spruce trees", "gold_answer": "Norway Spruce", "docid": "D1152040", "source": "official"},
    {"query": "how long does it take to get your bsrn if you already have a bachelors degree", "gold_answer": "takes approximately 4 years", "docid": "D51896", "source": "official"},
    {"query": "what cause pain on your right side", "gold_answer": "appendicitis, gallstones, kidney stones", "docid": "D522714", "source": "official"},
    {"query": "what degree do you need to become a nurse", "gold_answer": "have a Master", "docid": "D41976", "source": "official"},
    {"query": "what event started world war 1", "gold_answer": "assassination of Franz Ferdinand", "docid": "D226136", "source": "official"},
    {"query": "what is philosophical anthropology", "gold_answer": "what it means to be human", "docid": "D2564926", "source": "official"},
    {"query": "what is the test for cholecystitis", "gold_answer": "abdominal ultrasound", "docid": "D2966416", "source": "official"},
    {"query": "what is the weather like in jamaica in january", "gold_answer": "warm and balmy", "docid": "D2803937", "source": "official"},
    {"query": "what is the cause for tectonic plates", "gold_answer": "convection currents", "docid": "D225914", "source": "official"},
    {"query": "what is the best way to treat a cold sore that is scabbed over", "gold_answer": "phenol and menthol", "docid": "D222544", "source": "official"},
    {"query": "what are the two major subdivisions of the nervous system?", "gold_answer": "central and peripheral nervous systems", "docid": "D275769", "source": "official"},
    {"query": "apria healthcare npi number", "gold_answer": "1609166958", "docid": "D3248046", "source": "official"},
]

# 2. 직접 만든 질문
manual_queries = [
    {"query": "How many sides does a quadrilateral have?", "gold_answer": "four", "docid": "D2156914", "source": "manual"},
    {"query": "What country does raclette cheese originate from?", "gold_answer": "Switzerland", "docid": "D2304387", "source": "manual"},
    {"query": "How many times their own length can fleas jump?", "gold_answer": "150 times", "docid": "D675842", "source": "manual"},
    {"query": "What is the largest recorded length a goldfish can grow to?", "gold_answer": "23 inches", "docid": "D683584", "source": "manual"},
    {"query": "What causes most earthquakes according to plate tectonics?", "gold_answer": "the boundaries between the plates grind against each other", "docid": "D96127", "source": "manual"},
    {"query": "How many pairs of parallel sides does a square have?", "gold_answer": "two pairs", "docid": "D2682313", "source": "manual"},
    {"query": "When was Woodrow Wilson born?", "gold_answer": "December 28, 1856", "docid": "D330815", "source": "manual"},
    {"query": "How much does 3000 PSI concrete cost per yard?", "gold_answer": "$99.00 PER YARD", "docid": "D250659", "source": "manual"},
    {"query": "How long do fleas survive without a host?", "gold_answer": "a few days to two weeks", "docid": "D826377", "source": "manual"},
    {"query": "What is Earth's outer shell divided into according to plate tectonics?", "gold_answer": "several plates that glide over the mantle", "docid": "D2609667", "source": "manual"},
    {"query": "What is the small gap between two neurons called that neurotransmitters cross?", "gold_answer": "synapse", "docid": "D3117388", "source": "manual"},
    {"query": "What is the formula for the area of a trapezoid?", "gold_answer": "multiply the sum by the height", "docid": "D7862", "source": "manual"},
    {"query": "What percentage of the globe do oceans cover?", "gold_answer": "71 percent", "docid": "D155941", "source": "manual"},
    {"query": "What is the conservation status of the longfin mako shark?", "gold_answer": "Vulnerable", "docid": "D3558925", "source": "manual"},
    {"query": "Grey matter is what percentage of the brain?", "gold_answer": "40 percent", "docid": "D499866", "source": "manual"},
    {"query": "What family does the spruce tree belong to?", "gold_answer": "Pinaceae", "docid": "D804064", "source": "manual"},
]

# 3. 전체 합치기
test_queries_retrieval = official_queries + manual_queries

# 4. 생성(사람 채점) 평가용은 시간 제약 고려해 일부만 사용
test_queries_generation = test_queries_retrieval[:15]

if __name__ == "__main__":
    print("Retrieval 평가용 개수:", len(test_queries_retrieval))
    print("Generation 평가용 개수:", len(test_queries_generation))
