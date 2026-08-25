"""OpenAI-compatible local Qwen client shared by MathMD pipelines."""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8085/v1"
DEFAULT_MODEL_NAME = "Qwen3.8-27B-MLX-Mixed-4.95bpw"
DEFAULT_MODEL = str(Path.home() / ".omlx" / "models" / DEFAULT_MODEL_NAME)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def image_mime_type(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".heic":
        return "image/heic"
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


@dataclass(frozen=True)
class QwenConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    timeout: int = 300
    retries: int = 3


def resolve_qwen_config(
    env: Mapping[str, str] | None = None,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key_env: str = "QWEN_API_KEY",
    timeout: int = 300,
    retries: int = 3,
) -> QwenConfig:
    env = os.environ if env is None else env
    resolved_base = (
        base_url
        or env.get("QWEN_BASE_URL")
        or env.get("VIDEO2MD_BASE_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/")
    resolved_model = model or env.get("QWEN_MODEL") or env.get("VIDEO2MD_MODEL") or DEFAULT_MODEL
    api_key = env.get(api_key_env) if api_key_env else None
    if not api_key:
        api_key = env.get("QWEN_API_KEY")
    return QwenConfig(
        base_url=resolved_base,
        model=resolved_model,
        api_key=api_key,
        timeout=max(1, int(timeout)),
        retries=max(1, int(retries)),
    )


@dataclass
class QwenClient:
    config: QwenConfig
    request_post: Callable[..., object] = field(default=requests.post, repr=False)
    sleeper: Callable[[float], None] = field(default=time.sleep, repr=False)

    def complete(
        self,
        prompt_or_messages: str | list[dict[str, object]],
        *,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> str:
        messages = (
            [{"role": "user", "content": prompt_or_messages}]
            if isinstance(prompt_or_messages, str)
            else prompt_or_messages
        )
        return self._chat(messages, max_tokens=max_tokens, temperature=temperature)

    def vision(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> str:
        content: list[dict[str, object]] = []
        for raw_path in image_paths:
            path = Path(raw_path)
            encoded = base64.standard_b64encode(path.read_bytes()).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_mime_type(path)};base64,{encoded}"},
                }
            )
        content.append({"type": "text", "text": prompt})
        return self._chat(
            [{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _chat(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        last_error = "unknown local Qwen error"
        for attempt in range(self.config.retries):
            try:
                response = self.request_post(
                    url=f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers=auth_headers(self.config.api_key),
                    json={
                        "model": self.config.model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": messages,
                    },
                    timeout=self.config.timeout,
                )
            except Exception as exc:
                last_error = f"local Qwen request failed: {exc}"
                if attempt + 1 < self.config.retries:
                    self.sleeper(2**attempt)
                    continue
                break

            if response.status_code == 200:
                try:
                    content = response.json()["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"local Qwen returned an invalid response: {exc}") from exc
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("local Qwen returned empty content")
                return content

            body = (getattr(response, "text", "") or "")[:500].replace("\n", " ")
            last_error = f"local Qwen HTTP {response.status_code}: {body}"
            if response.status_code in RETRYABLE_STATUS_CODES and attempt + 1 < self.config.retries:
                self.sleeper(2**attempt)
                continue
            break
        raise RuntimeError(last_error)
