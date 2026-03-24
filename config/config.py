"""MLADE Configuration - reads from config.env file."""

import os
from dotenv import load_dotenv

_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_dir, "config.env"))

# ============================================================
# API (Alibaba Cloud Dashscope) - Target LLM
# ============================================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# ============================================================
# Google Cloud Translation API (Stage 1: question translation)
# ============================================================
GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY", "")

# ============================================================
# Voyage AI API (Stage 2: cross-lingual embeddings)
# ============================================================
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_EMBEDDING_MODEL = os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-multilingual-2")

# ============================================================
# Model IDs
# ============================================================
TARGET_LLM_MODEL = os.getenv("TARGET_LLM_MODEL", "qwen2.5-7b-instruct")

# ============================================================
# Languages (4 different scripts)
# ============================================================
LANGUAGES = {
    "en": {"google_code": "en", "name": "English"},
    "ar": {"google_code": "ar", "name": "Arabic"},
    "ja": {"google_code": "ja", "name": "Japanese"},
    "ru": {"google_code": "ru", "name": "Russian"},
}

SYSTEM_PROMPTS = {
    "en": "Answer with the key fact only. Be as brief as possible — one phrase or sentence maximum. Use digits for numbers (e.g. '2' not 'two'). No explanation.",
    "ar": "أجب بالحقيقة الأساسية فقط. كن موجزاً قدر الإمكان — جملة واحدة كحد أقصى. استخدم الأرقام للأعداد (مثل '2' وليس 'اثنان'). بدون شرح.",
    "ja": "重要な事実だけを答えてください。できる限り簡潔に——最大一文。数字はアラビア数字で表記してください（例：'2'）。説明は不要です。",
    "ru": "Отвечайте только ключевым фактом. Будьте максимально краткими — одна фраза или предложение. Используйте цифры для чисел (например '2', а не 'два'). Без объяснений.",
}

# ============================================================
# Stage 2: Consistency threshold (tau)
# ============================================================
CONSISTENCY_THRESHOLD = float(os.getenv("CONSISTENCY_THRESHOLD", "0.75"))

# ============================================================
# LLM Generation params
# ============================================================
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# ============================================================
# Stage 3: RAG Verification
# ============================================================
NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
WIKI_INDEX_DIR = os.path.join(_dir, os.getenv("WIKI_INDEX_DIR", "data/wiki_index"))
WIKI_TOP_K = int(os.getenv("WIKI_TOP_K", "5"))

# ============================================================
# Jina Embeddings V3 (Stage 3: fine retrieval per claim)
# ============================================================
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
