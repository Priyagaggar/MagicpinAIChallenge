"""Minimal multi-provider LLM chat client (raw HTTP, no SDK deps).

Configured entirely via environment variables:
    LLM_PROVIDER  - one of: openai, anthropic, gemini, deepseek, groq, openrouter
    LLM_API_KEY   - API key for the chosen provider
    LLM_MODEL     - model id (optional; each provider has a sane default)
"""
from __future__ import annotations

import os
from typing import Optional

import requests

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.0-flash",
    "deepseek": "deepseek-chat",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "anthropic/claude-3-haiku",
}

# OpenAI-compatible chat/completions providers (only base URL differs).
OPENAI_COMPAT_BASE_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


class LLMError(RuntimeError):
    pass


def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "openai").strip().lower()


def _api_key() -> str:
    return os.environ.get("LLM_API_KEY", "").strip()


def _model() -> str:
    provider = _provider()
    return os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODELS.get(provider, "")


def _call_openai_compat(base_url: str, system_prompt: str, user_prompt: str,
                         temperature: float, max_tokens: int, timeout: float,
                         extra: Optional[dict] = None) -> str:
    payload = {
        "model": _model(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    payload.update(extra or {})
    resp = requests.post(
        base_url,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content") or ""
    if not content.strip():
        raise LLMError(f"Empty content in LLM response (finish_reason="
                        f"{data['choices'][0].get('finish_reason')!r}); response was truncated "
                        f"before any output — increase max_tokens or lower reasoning effort.")
    return content


def _call_anthropic(system_prompt: str, user_prompt: str,
                     temperature: float, max_tokens: int, timeout: float) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": _api_key(),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": _model(),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(block.get("text", "") for block in data.get("content", []))


def _call_gemini(system_prompt: str, user_prompt: str,
                  temperature: float, max_tokens: int, timeout: float) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{_model()}:generateContent"
        f"?key={_api_key()}"
    )
    resp = requests.post(
        url,
        json={
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def chat(system_prompt: str, user_prompt: str, *,
         temperature: float = 0.0, max_tokens: int = 1200, timeout: float = 20.0) -> str:
    """Send one system+user turn to the configured LLM. Returns raw text content."""
    provider = _provider()
    if not _api_key():
        raise LLMError(f"LLM_API_KEY is not set for provider '{provider}'")

    if provider in OPENAI_COMPAT_BASE_URLS:
        extra = {}
        if provider == "groq":
            # groq's gpt-oss models spend max_tokens on hidden reasoning first; capping
            # reasoning effort keeps the actual JSON reply from getting truncated away.
            extra["reasoning_effort"] = os.environ.get("LLM_REASONING_EFFORT", "low")
        return _call_openai_compat(OPENAI_COMPAT_BASE_URLS[provider], system_prompt, user_prompt,
                                    temperature, max_tokens, timeout, extra)
    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_prompt, temperature, max_tokens, timeout)
    if provider == "gemini":
        return _call_gemini(system_prompt, user_prompt, temperature, max_tokens, timeout)

    raise LLMError(f"Unknown LLM_PROVIDER: {provider}")
