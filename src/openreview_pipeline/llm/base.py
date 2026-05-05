from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from threading import BoundedSemaphore
import threading
import time
from typing import Any, Callable, Optional

import argparse
import json
import yaml

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_json(self, prompt: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def generate_with_pdf_url(self, prompt: str, pdf_url: str, **kwargs) -> str:
        raise NotImplementedError


class MockLLMBackend(LLMBackend):
    def __init__(self, response: str = "mock response"):
        self._response = response

    def generate(self, prompt: str, **kwargs) -> str:
        return self._response

    def generate_json(self, prompt: str, **kwargs) -> dict[str, Any]:
        return {"result": self._response}

    def generate_with_pdf_url(self, prompt: str, pdf_url: str, **kwargs) -> str:
        return self._response


@dataclass
class _LLMKeySlot:
    client: Any
    masked_token: str
    semaphore: BoundedSemaphore
    lock: threading.Lock
    last_request_started_at: float = 0.0


class LLMRequestManager:
    def __init__(
        self,
        *,
        base_url: str,
        api_tokens: list[str],
        per_key_request_interval_seconds: float = 0.0,
        per_key_max_concurrent_requests: int = 1,
        max_retries: int = 3,
        retry_backoff_seconds: float = 8.0,
    ):
        from openai import OpenAI

        cleaned_tokens = [str(token).strip() for token in api_tokens if str(token).strip()]
        if not cleaned_tokens:
            raise ValueError("llm.api_tokens must contain at least one API token")

        self.base_url = base_url.rstrip("/")
        self.per_key_request_interval_seconds = max(0.0, float(per_key_request_interval_seconds))
        self.per_key_max_concurrent_requests = max(1, int(per_key_max_concurrent_requests))
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._slots = [
            _LLMKeySlot(
                client=OpenAI(api_key=token, base_url=self.base_url),
                masked_token=self._mask_token(token),
                semaphore=BoundedSemaphore(self.per_key_max_concurrent_requests),
                lock=threading.Lock(),
            )
            for token in cleaned_tokens
        ]
        self._lock = threading.Lock()
        self._next_slot_index = 0

    @staticmethod
    def _mask_token(token: str) -> str:
        token = token.strip()
        if len(token) <= 4:
            return "****"
        return f"...{token[-4:]}"

    def _ordered_slots(self, avoid_slot: Optional[_LLMKeySlot] = None) -> list[_LLMKeySlot]:
        with self._lock:
            start_index = self._next_slot_index
            self._next_slot_index = (self._next_slot_index + 1) % len(self._slots)
            ordered = [
                self._slots[(start_index + offset) % len(self._slots)]
                for offset in range(len(self._slots))
            ]
            if avoid_slot is not None and len(self._slots) > 1:
                ordered = [slot for slot in ordered if slot is not avoid_slot]
            return ordered or list(self._slots)

    def _acquire_slot(self, avoid_slot: Optional[_LLMKeySlot] = None) -> _LLMKeySlot:
        ordered = self._ordered_slots(avoid_slot=avoid_slot)
        for slot in ordered:
            if slot.semaphore.acquire(blocking=False):
                return slot

        slot = ordered[0]
        slot.semaphore.acquire()
        return slot

    def _wait_for_slot_interval(self, slot: _LLMKeySlot) -> None:
        with slot.lock:
            if self.per_key_request_interval_seconds > 0 and slot.last_request_started_at > 0:
                elapsed = time.monotonic() - slot.last_request_started_at
                sleep_seconds = self.per_key_request_interval_seconds - elapsed
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            slot.last_request_started_at = time.monotonic()

    def call(self, operation_name: str, fn: Callable[[Any], Any]) -> Any:
        last_exc = None
        failed_slot: Optional[_LLMKeySlot] = None
        for attempt in range(1, self.max_retries + 1):
            slot = self._acquire_slot(avoid_slot=failed_slot)
            retry_sleep_seconds: Optional[float] = None
            try:
                self._wait_for_slot_interval(slot)
                return fn(slot.client)
            except Exception as exc:
                last_exc = exc
                failed_slot = slot
                if attempt >= self.max_retries:
                    break
                sleep_seconds = self.retry_backoff_seconds * attempt
                logger.warning(
                    "%s failed with API key %s on attempt %s/%s: %s. Retrying in %.1fs.",
                    operation_name,
                    slot.masked_token,
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_seconds,
                )
                retry_sleep_seconds = sleep_seconds
            finally:
                slot.semaphore.release()
            if retry_sleep_seconds is not None:
                time.sleep(retry_sleep_seconds)
        raise last_exc


class OpenAICompatibleBackend(LLMBackend):
    def __init__(
        self,
        base_url: str,
        api_tokens: list[str],
        model: str = "gpt-4o-mini",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        per_key_request_interval_seconds: float = 0.0,
        per_key_max_concurrent_requests: int = 1,
        max_retries: int = 3,
        retry_backoff_seconds: float = 8.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.request_manager = LLMRequestManager(
            base_url=base_url,
            api_tokens=api_tokens,
            per_key_request_interval_seconds=per_key_request_interval_seconds,
            per_key_max_concurrent_requests=per_key_max_concurrent_requests,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
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
        response = self.request_manager.call(
            "chat completion",
            lambda client: client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **self._chat_completion_kwargs(),
            ),
        )

        # pretty_print_response(response)
        return response.choices[0].message.content

    def response(self, prompt: str, **kwargs) -> Any:
        response = self.request_manager.call(
            "responses completion",
            lambda client: client.responses.create(
                input=prompt,
                **self._responses_kwargs(),
            ),
        )
        pretty_print_response(response)
        return response

    def generate_json(self, prompt: str, **kwargs) -> dict[str, Any]:
        response = self.request_manager.call(
            "chat JSON completion",
            lambda client: client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **self._chat_completion_kwargs(),
            ),
        )
        content = response.choices[0].message.content
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            return json.loads(json_match.group())
        return {"raw": content}

    def _extract_response_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        try:
            response_dict = response.to_dict()
        except Exception:
            response_dict = getattr(response, "__dict__", {})

        if isinstance(response_dict, dict):
            output = response_dict.get("output") or []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content") or []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text = block.get("text")
                        if text:
                            return str(text)
        return ""

    def build_pdf_response_payload(self, prompt: str, file_input: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        file_input,
                    ],
                }
            ],
        }
        payload.update(self._responses_kwargs())
        return payload

    def generate_with_pdf_url(
        self,
        prompt: str,
        pdf_url: str,
        *,
        debug: bool = False,
        **kwargs,
    ) -> str:
        payload = self.build_pdf_response_payload(
            prompt,
            {
                "type": "input_file",
                "file_url": pdf_url,
            },
        )

        if debug:
            print("responses_payload:")
            print(json.dumps(payload, indent=2, default=str))

        response = self.request_manager.call(
            "PDF URL response completion",
            lambda client: client.responses.create(**payload),
        )

        if debug:
            print("responses_raw:")
            pretty_print_response(response)

        return self._extract_response_text(response)


def load_llm_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    llm_config = config.get("llm", {})
    if not isinstance(llm_config, dict):
        raise ValueError("config.yaml 'llm' section must be a mapping")

    required_keys = ["base_url", "api_tokens", "model"]
    missing = [key for key in required_keys if not llm_config.get(key)]
    if missing:
        raise ValueError(f"Missing llm config keys: {', '.join(missing)}")
    if not isinstance(llm_config["api_tokens"], list) or not llm_config["api_tokens"]:
        raise ValueError("llm.api_tokens must be a non-empty list")

    return llm_config


def create_openai_compatible_backend(
    *,
    base_url: str,
    api_tokens: list[str],
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    per_key_request_interval_seconds: float = 0.0,
    per_key_max_concurrent_requests: int = 1,
    max_retries: int = 3,
    retry_backoff_seconds: float = 8.0,
) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(
        base_url=base_url,
        api_tokens=api_tokens,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        per_key_request_interval_seconds=per_key_request_interval_seconds,
        per_key_max_concurrent_requests=per_key_max_concurrent_requests,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )


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
    parser.add_argument(
        "--debug-pdf",
        action="store_true",
        help="Print uploaded file metadata, request payload, and raw response for PDF tests",
    )
    parser.add_argument(
        "--pdf-url",
        default=None,
        help="Optional remote PDF URL for testing direct file_url input through the Responses API",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    llm_config = load_llm_config(config_path)

    backend = OpenAICompatibleBackend(
        base_url=llm_config["base_url"],
        api_tokens=llm_config["api_tokens"],
        model=llm_config["model"],
        max_tokens=int(llm_config.get("max_tokens", 4096)),
        temperature=float(llm_config.get("temperature", 0.0)),
        per_key_request_interval_seconds=float(llm_config.get("per_key_request_interval_seconds", 0.0)),
        per_key_max_concurrent_requests=int(llm_config.get("per_key_max_concurrent_requests", 1)),
        max_retries=int(llm_config.get("max_retries", 3)),
        retry_backoff_seconds=float(llm_config.get("retry_backoff_seconds", 8.0)),
    )

    print(f"config: {config_path}")
    print(f"model: {llm_config['model']}")
    print(f"base_url: {llm_config['base_url']}")

    if args.pdf_url:
        result = backend.generate_with_pdf_url(
            args.prompt,
            args.pdf_url,
            debug=args.debug_pdf,
        )
    elif args.json:
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
