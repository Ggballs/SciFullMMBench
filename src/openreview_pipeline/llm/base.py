from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import argparse
import yaml


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_json(self, prompt: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError


class MockLLMBackend(LLMBackend):
    def __init__(self, response: str = "mock response"):
        self._response = response

    def generate(self, prompt: str, **kwargs) -> str:
        return self._response

    def generate_json(self, prompt: str, **kwargs) -> dict[str, Any]:
        return {"result": self._response}


class OpenAICompatibleBackend(LLMBackend):
    def __init__(
        self,
        base_url: str,
        api_token: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        from openai import OpenAI
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self._client = OpenAI(
            api_key=api_token,
            base_url=base_url,
        )

    def _chat_completion_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs

    def _responses_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **self._chat_completion_kwargs(),
        )

        pretty_print_response(response)
        return response.choices[0].message.content

    def response(self, prompt: str, **kwargs) -> Any:
        response = self._client.responses.create(
                input=prompt,
                **self._responses_kwargs(),
        )
        pretty_print_response(response)
        return response

    def generate_json(self, prompt: str, **kwargs) -> dict[str, Any]:
        import json
        import re
        response = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **self._chat_completion_kwargs(),
        )
        content = response.choices[0].message.content
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            return json.loads(json_match.group())
        return {"raw": content}


def _load_llm_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    llm_config = config.get("llm", {})
    if not isinstance(llm_config, dict):
        raise ValueError("config.yaml 'llm' section must be a mapping")

    required_keys = ["base_url", "api_token", "model"]
    missing = [key for key in required_keys if not llm_config.get(key)]
    if missing:
        raise ValueError(f"Missing llm config keys: {', '.join(missing)}")

    return llm_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the configured LLM backend.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[3] / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--prompt",
        default="Who are you and your version",
        help="Prompt to send to the configured model",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Call generate_json instead of generate",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    llm_config = _load_llm_config(config_path)

    backend = OpenAICompatibleBackend(
        base_url=llm_config["base_url"],
        api_token=llm_config["api_token"],
        model=llm_config["model"],
    )

    print(f"config: {config_path}")
    print(f"model: {llm_config['model']}")
    print(f"base_url: {llm_config['base_url']}")

    if args.json:
        result = backend.generate_json(args.prompt)
    # elif backend.model.startswith("gpt-5.4"):
    #     result = backend.completion(args.prompt)
    else:
        result = backend.generate(args.prompt)

    print("result:")
    print(result)

import json

def pretty_print_response(response):
    # Try common conversions, fall back to __dict__/str
    try:
        body = response.to_dict()  # new OpenAI SDK often supports this
    except Exception:
        try:
            body = response.json()  # sometimes available
        except Exception:
            try:
                body = dict(response)  # works if response is mapping-like
            except Exception:
                body = getattr(response, "__dict__", str(response))
    print(json.dumps(body, indent=2, default=str))

if __name__ == "__main__":
    main()
