"""생성 LLM 호출 (OpenAI 호환 chat API — 기본 GPT-4o-mini).

로컬 모델 로드 없이 API만 호출한다. 설정은 환경변수로:
  OPENAI_API_KEY : API 키 (필수)
  LLM_API_URL    : 기본 https://api.openai.com/v1 (호환 서버로 교체 가능)
  LLM_MODEL      : 기본 gpt-4o-mini (저비용 생성 담당)

temperature 기본 0 — 결정적 응답으로 불필요한 재시도를 막는다(비용 절약).
"""

from __future__ import annotations

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()   # import 순서와 무관하게 .env 를 먼저 읽는다

LLM_API_URL = os.getenv("LLM_API_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def chat(system: str, user: str, max_tokens: int = 512,
         temperature: float = 0.0, model: str | None = None) -> str:
    """system/user 프롬프트로 1회 응답 생성. 429/5xx는 지수 백오프로 3회 재시도."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY 가 비어 있습니다. .env 에 키를 넣어주세요.")
    payload = {
        "model": model or LLM_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }
    for attempt in range(3):
        r = requests.post(f"{LLM_API_URL}/chat/completions",
                          headers={"Authorization": f"Bearer {key}"},
                          json=payload, timeout=120)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)   # ponytail: 고정 3회 백오프, 부족하면 tenacity 도입
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    r.raise_for_status()   # 재시도 소진 — 마지막 응답의 에러를 그대로 올린다
    raise RuntimeError("unreachable")
