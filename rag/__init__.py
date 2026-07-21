"""패키지 임포트 시점에 .env 를 한 번 읽는다 (CLI든 Streamlit이든 동일하게 적용)."""

from dotenv import load_dotenv

load_dotenv()
