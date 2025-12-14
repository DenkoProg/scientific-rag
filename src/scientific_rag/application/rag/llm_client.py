import os
from typing import Any

from litellm import completion
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# Assuming scientific_rag.settings is available in your project structure
from scientific_rag.settings import settings


class LLMClient:
    def __init__(self):
        self.model = settings.llm_model
        self.api_key = settings.llm_api_key
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

        if not self.api_key:
            logger.warning(f"No API key provided for model {self.model}. Ensure this is intended.")

    # Increased max wait to 20s to handle OpenRouter's "free tier" congestion better
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=20), reraise=True)
    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            logger.debug(f"Sending request to {self.model} via OpenRouter")

            response = completion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                custom_llm_provider="openai",
                api_base="https://openrouter.ai/api/v1",
                extra_headers={
                    "HTTP-Referer": "http://localhost:8000",  # TODO: Replace with your app's URL
                    "X-Title": "Scientific RAG",
                },
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )

            content = response.choices[0].message.content
            return content.strip()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise e


llm_client = LLMClient()
