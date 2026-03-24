"""Google Cloud Translation API - translates questions across languages.

Replaces local NLLB-200. Free tier: 500,000 chars/month.
API docs: https://cloud.google.com/translate/docs/basic/translating-text
"""

import json
import time
import urllib.request
import urllib.parse
import urllib.error
import config


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate text from src_lang to tgt_lang using Google Cloud Translation API.

    Args:
        text: Input text.
        src_lang: Source language code (e.g., "en").
        tgt_lang: Target language code (e.g., "ar").

    Returns:
        Translated text.
    """
    src_code = config.LANGUAGES[src_lang]["google_code"]
    tgt_code = config.LANGUAGES[tgt_lang]["google_code"]

    url = "https://translation.googleapis.com/language/translate/v2"
    params = urllib.parse.urlencode({
        "q": text,
        "source": src_code,
        "target": tgt_code,
        "format": "text",
        "key": config.GOOGLE_TRANSLATE_API_KEY,
    })

    req = urllib.request.Request(
        f"{url}?{params}",
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["data"]["translations"][0]["translatedText"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                wait = 2 ** attempt  # 1s, 2s
                print(f"  Google Translate 429, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def translate_to_all(text: str, src_lang: str = "en") -> dict[str, str]:
    """Translate text from src_lang to all other target languages.

    Returns:
        Dict mapping lang code -> translated text.
        Includes the original text under src_lang.
    """
    results = {src_lang: text}
    for lang_code in config.LANGUAGES:
        if lang_code == src_lang:
            continue
        results[lang_code] = translate(text, src_lang, lang_code)
        time.sleep(0.3)  # avoid hitting Google Translate rate limit
    return results
