"""
hsi_research_agent/llm/client.py
Local Ollama client with strict Pydantic structured output validation, retry, and repair prompts.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Type, TypeVar
import requests
from pydantic import BaseModel, ValidationError

from hsi_research_agent.config import settings

logger = logging.getLogger("hsi_research_agent.llm")

T = TypeVar("T", bound=BaseModel)


class OllamaClient:
    """Interface to local Ollama server running Qwen 2.5 14B."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

    def check_connection(self) -> Dict[str, Any]:
        """
        Verify Ollama service is reachable and the required model is installed.
        Returns model details dictionary or raises RuntimeError.
        """
        url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama server responded with status {resp.status_code} at {self.base_url}"
                )
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}: {str(e)}.\n"
                f"Please ensure Ollama is running ('ollama serve')."
            ) from e

        data = resp.json()
        models = data.get("models", [])
        installed_names = [m.get("name", "") for m in models]

        # Match exact name first
        matched = None
        for m in models:
            name = m.get("name", "")
            if name == self.model:
                matched = m
                break

        # Fallback to closest matching variant or installed qwen2.5 model
        if not matched:
            for m in models:
                name = m.get("name", "")
                if (
                    name.startswith(f"{self.model}:")
                    or self.model.split(":")[0] == name.split(":")[0]
                ):
                    matched = m
                    logger.info("Model '%s' mapped to installed variant '%s'.", self.model, name)
                    self.model = name
                    break

        if not matched and models:
            # Fall back to any installed model
            matched = models[0]
            logger.warning("Configured model '%s' not found. Falling back to '%s'.", self.model, matched.get("name"))
            self.model = matched.get("name")

        if not matched:
            raise RuntimeError(
                f"No models found in Ollama.\n"
                f"Please download one via: ollama pull qwen2.5:14b"
            )

        logger.info("Connected to Ollama with model: %s", self.model)
        return matched

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are an expert scientific researcher.",
        temperature: float = 0.2,
        num_ctx: int = 8192,
    ) -> str:
        """Execute a text generation call via Ollama /api/chat."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }

        resp = requests.post(url, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama call failed with status {resp.status_code}: {resp.text}"
            )

        msg = resp.json().get("message", {}).get("content", "")
        return msg.strip()

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are an expert scientific researcher.",
        temperature: float = 0.2,
        num_ctx: int = 8192,
    ) -> str:
        """Alias for generate()."""
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            num_ctx=num_ctx,
        )

    def generate_structured(
        self,
        prompt: str,
        pydantic_class: Type[T],
        system_prompt: str = "You are an expert scientific literature analyst. Output strict valid JSON only.",
        temperature: float = 0.1,
        max_retries: int = 3,
        num_ctx: int = 8192,
    ) -> T:
        """
        Generate a structured response parsed and validated against a Pydantic model.
        On invalid JSON or validation error, executes a repair prompt with feedback.
        Never silently accepts malformed data.
        """
        schema_json = json.dumps(pydantic_class.model_json_schema(), indent=2)
        augmented_prompt = (
            f"{prompt}\n\n"
            f"You MUST return strict, valid JSON matching this exact JSON Schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Output JSON ONLY without explanation or commentary."
        )

        current_prompt = augmented_prompt
        last_error = ""

        for attempt in range(max_retries):
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                },
            }

            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
                    time.sleep(1.0)
                    continue

                raw_content = resp.json().get("message", {}).get("content", "").strip()
                # Clean code blocks
                clean_text = re.sub(
                    r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.MULTILINE
                ).strip()

                parsed_json = json.loads(clean_text)
                validated = pydantic_class.model_validate(parsed_json)
                return validated

            except (json.JSONDecodeError, ValidationError, Exception) as e:
                last_error = f"{type(e).__name__}: {str(e)}"
                logger.warning(
                    "Structured generation attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    max_retries,
                    pydantic_class.__name__,
                    last_error,
                )
                # Repair prompt for next attempt
                current_prompt = (
                    f"Your previous JSON output failed validation with error: {last_error}.\n"
                    f"Please correct the JSON according to the schema:\n"
                    f"```json\n{schema_json}\n```\n"
                    f"Original instructions:\n{prompt}"
                )
                time.sleep(1.0)

        raise ValueError(
            f"Failed to generate valid structured data for {pydantic_class.__name__} "
            f"after {max_retries} attempts. Last error: {last_error}"
        )

    def invoke_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: str = "You are an expert scientific literature analyst. Output strict valid JSON only.",
        temperature: float = 0.1,
        max_retries: int = 3,
        num_ctx: int = 8192,
    ) -> T:
        """Alias for generate_structured() accepting response_model parameter."""
        return self.generate_structured(
            prompt=prompt,
            pydantic_class=response_model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_retries=max_retries,
            num_ctx=num_ctx,
        )

