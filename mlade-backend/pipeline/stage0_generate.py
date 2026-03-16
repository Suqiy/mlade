"""Stage 0: Original Response Generation.

Target LLM (bare model, no RAG) answers the question directly.
This response is what we will audit for hallucination.
"""

from models import target_llm
import config


def run(question: str, src_lang: str = "en") -> str:
    """Generate original response from Target LLM.

    Args:
        question: The factual question.
        src_lang: Language of the question.

    Returns:
        The LLM's original response (the object of our audit).
    """
    system_prompt = config.SYSTEM_PROMPTS.get(src_lang, config.SYSTEM_PROMPTS["en"])
    return target_llm.generate(question, system_prompt=system_prompt)
