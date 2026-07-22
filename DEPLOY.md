# 배포 가이드 (데스크탑 RTX 3060 + Docker + Cloudflare Tunnel)

로컬 서브셋(`data/ko_miracl_reduced_corpus.jsonl`, 20만 청크 / 문서 31,514개)을
bge-m3로 임베딩해 영속 ChromaDB에 적재하고, 검색 API를 Cloudflare Tunnel로 공개한다.

```
로컬 서브셋(jsonl) ──ingest──▶ [chroma: 영속 볼륨]
                                      ▲
                         [api: bge-m3 임베딩 + FastAPI /search] ── GPU
                                      ▲
                         [cloudflared] ──▶ https://<이름>.trycloudflare 또는 커스텀 도메인
```

## 0. 사전 준비 (데스크탑, 1회)

- Docker Desktop + **NVIDIA Container Toolkit** (GPU 패스스루). 확인:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
  ```
- `data/ko_miracl_reduced_corpus.jsonl` 이 `data/` 에 있어야 한다.
- Cloudflare Zero Trust 대시보드 → Networks → Tunnels → **Create tunnel** →
  커넥터 **Docker** 선택 → 표시되는 토큰(`eyJ...`) 복사.
  같은 화면 **Public Hostname** 에서 서비스를 `http://api:8080` 으로 라우팅.
- 환경변수 파일:
  ```bash
  cp .env.example .env      # TUNNEL_TOKEN 붙여넣기
  ```

## 1. 벡터 저장소 기동

```bash
docker compose up -d chroma
```

## 2. 서브셋 적재 (1회, 20만 청크)

```bash
docker compose run --rm api python -m rag.ingest --model bge-m3
```

- RTX 3060 기준 대략 수십 분(임베딩이 병목). 진행률/ETA가 로그로 나온다.
- 다시 실행해도 id 기준 **upsert** 라 중복되지 않는다. 처음부터 다시 하려면 `--reset`.
- 파이프라인만 빠르게 점검하려면 `--limit 2000`.

## 3. API + 터널 공개

```bash
docker compose up -d api cloudflared
```

확인:
```bash
curl http://localhost:8080/health
curl "http://localhost:8080/search?q=우주왕복선 STS-133 임무&k=5"
```

Cloudflare가 발급한 공개 URL로도 동일하게 `/search` 호출 가능.

## 임베딩 모델 교체 / 비교

`rag/config.py` 의 `EMBED_MODELS` 에 등록된 key 로 고른다 (기본 `bge-m3`).
컬렉션이 `komiracl_{model}` 로 분리되므로 여러 모델을 나란히 올려 비교할 수 있다.

```bash
# 다른 후보를 별도 컬렉션에 적재
docker compose run --rm -e EMBED_MODEL=ko-sroberta api python -m rag.ingest --model ko-sroberta

# 그 모델로 API 띄우기
EMBED_MODEL=ko-sroberta docker compose up -d api
```

새 모델을 비교하려면 `EMBED_MODELS` 에 한 줄 추가만 하면 된다.

## 참고

- 평가(queries/qrels)는 여전히 HF `taeminlee/Ko-miracl` 에서 로드한다(정답셋).
  노트북/`rag.cli` 의 in-memory 비교 흐름을 이 로컬 서브셋으로 바꾸고 싶으면
  `rag/data.py` 의 `collect_corpus(...)` 대신 `load_local_corpus(path)` 를 쓰면 된다.
- Chroma를 로컬에서 직접 디버깅하려면 `docker-compose.yml` 의 chroma `ports` 주석 해제.
