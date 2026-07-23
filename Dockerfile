# RTX 3060(CUDA)에서 bge-m3 임베딩을 돌리는 검색 API 이미지.
# torch+cuda가 이미 포함된 공식 pytorch 런타임 이미지를 베이스로 쓴다.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# 의존성 먼저 (torch는 베이스에 있으므로 requirements에서 제외해도 되지만,
# 버전 고정을 위해 그대로 두고 --no-deps 없이 설치한다)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY rag ./rag

EXPOSE 8080

# 기본은 검색 API. 적재는 `docker compose run --rm api python -m rag.ingest` 로 실행.
CMD ["uvicorn", "rag.server:app", "--host", "0.0.0.0", "--port", "8080"]
