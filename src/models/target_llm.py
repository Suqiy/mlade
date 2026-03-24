"""Target LLM wrapper - calls Qwen2.5-7B via Alibaba Cloud Dashscope API."""

from openai import OpenAI
import config

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
        )
    return _client


def generate(question: str, system_prompt: str | None = None) -> str:
    """Generate a response from the Target LLM.

    Each call is independent (isolated) - no conversation history shared.
    """
    client = get_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=config.TARGET_LLM_MODEL,
        messages=messages,
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=config.LLM_TEMPERATURE,
    )

    return response.choices[0].message.content.strip()
