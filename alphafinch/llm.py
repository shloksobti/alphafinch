"""LLM providers. Every provider exposes `complete(system, prompt) -> str`.

  anthropic    Claude via the Anthropic API (ANTHROPIC_API_KEY or `ant auth login`)
  openai       OpenAI API (OPENAI_API_KEY)
  ollama       a local model through Ollama's OpenAI-compatible server (no key)
  compatible   any OpenAI-compatible server (--base-url, --model, optional key)
  claude-code  your existing Claude Code install (`claude -p`), using your subscription
  none         no LLM: only parameter tweaks and signal-blend crossover (offline demo)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-5",
    "ollama": "qwen2.5-coder:14b",
    "compatible": None,
    "claude-code": "sonnet",
}


class LLMError(Exception):
    pass


class Provider:
    name = "base"
    model = ""

    def complete(self, system: str, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model=None, effort="medium"):
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or DEFAULT_MODELS["anthropic"]
        self.effort = effort

    def complete(self, system, prompt):
        a = self._anthropic
        try:
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=16000, system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
        except a.RateLimitError as e:
            raise LLMError(f"rate limited: {e}") from None
        except a.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}: {e.message}") from None
        except a.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from None
        if resp.stop_reason == "refusal":
            raise LLMError("model declined the request")
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAICompatProvider(Provider):
    def __init__(self, name, model=None, base_url=None, api_key=None):
        import openai
        self._openai = openai
        self.name = name
        if name == "ollama":
            base_url = base_url or "http://localhost:11434/v1"
            api_key = api_key or "ollama"
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model or DEFAULT_MODELS.get(name)
        if not self.model:
            raise LLMError("--model is required for the compatible provider")

    def complete(self, system, prompt):
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            )
        except self._openai.OpenAIError as e:
            raise LLMError(str(e)) from None
        return r.choices[0].message.content or ""


class ClaudeCodeProvider(Provider):
    """Runs `claude -p` with every tool disabled: a pure text completion billed to the
    user's Claude subscription. No MCP servers, no settings, no session files."""
    name = "claude-code"

    def __init__(self, model=None, timeout=300):
        if not shutil.which("claude"):
            raise LLMError("Claude Code CLI not found (install it, or choose another provider)")
        self.model = model or DEFAULT_MODELS["claude-code"]
        self.timeout = timeout

    def complete(self, system, prompt):
        cmd = ["claude", "-p", prompt, "--model", self.model, "--system-prompt", system,
               "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
               "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise LLMError("claude -p timed out") from None
        try:
            data = json.loads(out.stdout)
        except json.JSONDecodeError:
            raise LLMError((out.stderr or out.stdout)[-300:]) from None
        if data.get("is_error"):
            raise LLMError(str(data.get("result"))[:300])
        return data.get("result", "")


def get(provider: str, model: str | None = None, base_url: str | None = None,
        api_key: str | None = None) -> Provider | None:
    if provider == "none":
        return None
    if provider == "anthropic":
        return AnthropicProvider(model)
    if provider in ("openai", "ollama", "compatible"):
        return OpenAICompatProvider(provider, model, base_url, api_key)
    if provider == "claude-code":
        return ClaudeCodeProvider(model)
    raise LLMError(f"unknown provider '{provider}'")


def auto() -> str:
    """Pick a provider from what is available on this machine."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if shutil.which("claude"):
        return "claude-code"
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
        return "ollama"
    except Exception:
        return "none"
