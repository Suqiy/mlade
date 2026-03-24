"""Stage 1: Multilingual Parallel Generation.

Translate question to 4 languages, then independently query the same
Target LLM in each language. Each call is isolated (no shared context).
"""

from models import target_llm, translator
import config


def run(question: str, src_lang: str = "en") -> dict[str, dict]:
    """Generate multilingual responses.

    Args:
        question: The original question.
        src_lang: Language of the original question.

    Returns:
        Dict mapping lang_code -> {"question": str, "response": str}.
    """
    # Step 1: Translate question to all target languages
    translated_questions = translator.translate_to_all(question, src_lang=src_lang)

    # Step 2: Independently generate response in each language
    results = {}
    for lang_code, q_translated in translated_questions.items():
        system_prompt = config.SYSTEM_PROMPTS[lang_code]
        response = target_llm.generate(q_translated, system_prompt=system_prompt)
        results[lang_code] = {
            "question": q_translated,
            "response": response,
        }

    return results
