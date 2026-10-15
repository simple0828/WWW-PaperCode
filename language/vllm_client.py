"""OpenAI-compatible LLM client for MMD-TFS (local vLLM or Zhipu GLM).

Default server: http://10.30.1.2:8000 (vLLM).

Also supports Zhipu OpenAI-compatible API::

  LLM_PROVIDER=zhipu
  ZAI_API_KEY=...                 # or ZHIPU_API_KEY / VLLM_API_KEY
  VLLM_MODEL=glm-5.3-flash        # free / lightest abundant pack
  VLLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4

Qwen3 on vLLM may put output in ``message.reasoning`` when thinking is enabled.
Set ``enable_thinking=False`` (default) to get direct answers in ``message.content``.
For Zhipu: ``thinking.type`` is ``enabled|disabled``; effort is separate
``reasoning_effort=low|high|max`` (``VLLM_REASONING_EFFORT``).
``glm-5.3-flash`` always thinks — never send ``disabled``; default effort ``low``.

Environment variables (all optional):
  LLM_PROVIDER    vllm | zhipu (auto if base_url contains bigmodel.cn)
  VLLM_BASE_URL   e.g. http://10.30.1.2:8000 or Zhipu paas/v4 URL
  VLLM_MODEL      e.g. Qwen/... or glm-5.3-flash
  VLLM_API_KEY / ZAI_API_KEY / ZHIPU_API_KEY
  VLLM_MAX_TOKENS output cap; unset = no cap (use server remaining context)
  VLLM_MAX_MODEL_LEN override model context window (default: from /v1/models)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

Message = Dict[str, str]

ZHIPU_DEFAULT_BASE = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_DEFAULT_MODEL = "glm-5.3-flash"


class OutputMode(str, Enum):
    """How to extract generated text from vLLM chat responses."""

    AUTO = "auto"
    CONTENT = "content"
    REASONING = "reasoning"


class VLLMError(RuntimeError):
    """Raised when the vLLM server returns an error or an unexpected payload."""


def _detect_provider(base_url: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.strip().lower()
    env = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if env in {"vllm", "zhipu", "zai", "glm"}:
        return "zhipu" if env in {"zhipu", "zai", "glm"} else "vllm"
    if "bigmodel.cn" in (base_url or "").lower():
        return "zhipu"
    return "vllm"


@dataclass
class VLLMConfig:
  base_url: str = "http://10.30.1.2:8000"
  model: Optional[str] = None
  api_key: Optional[str] = None
  timeout: float = 600.0
  enable_thinking: bool = False
  output_mode: OutputMode = OutputMode.AUTO
  default_temperature: float = 0.0
  # None = do not send max_tokens; vLLM uses full remaining context window.
  default_max_tokens: Optional[int] = None
  max_model_len: Optional[int] = None
  extra_body: Dict[str, Any] = field(default_factory=dict)
  # vllm | zhipu — controls chat path and thinking payload shape.
  provider: str = "vllm"

  @classmethod
  def from_env(cls) -> "VLLMConfig":
    mode = os.getenv("VLLM_OUTPUT_MODE", OutputMode.AUTO.value)
    try:
      output_mode = OutputMode(mode)
    except ValueError:
      output_mode = OutputMode.AUTO

    enable_thinking = os.getenv("VLLM_ENABLE_THINKING", "").lower() in {
      "1",
      "true",
      "yes",
    }

    max_tokens_env = os.getenv("VLLM_MAX_TOKENS")
    max_model_len_env = os.getenv("VLLM_MAX_MODEL_LEN")
    provider_hint = (os.getenv("LLM_PROVIDER") or "").strip().lower() or None
    default_base = (
        ZHIPU_DEFAULT_BASE
        if provider_hint in {"zhipu", "zai", "glm"}
        else "http://10.30.1.2:8001"
    )
    base_url = os.getenv("VLLM_BASE_URL", default_base).rstrip("/")
    provider = _detect_provider(base_url, provider_hint)
    api_key = (
        os.getenv("VLLM_API_KEY")
        or os.getenv("ZAI_API_KEY")
        or os.getenv("ZHIPU_API_KEY")
    )
    model = os.getenv("VLLM_MODEL")
    if provider == "zhipu" and not model:
      model = ZHIPU_DEFAULT_MODEL
    # Zhipu free packs: keep outputs short by default (cleaning / JSON).
    default_max_tokens = int(max_tokens_env) if max_tokens_env else (
        1024 if provider == "zhipu" else None
    )

    return cls(
      base_url=base_url,
      model=model,
      api_key=api_key,
      timeout=float(os.getenv("VLLM_TIMEOUT", "600")),
      enable_thinking=enable_thinking,
      output_mode=output_mode,
      default_temperature=float(os.getenv("VLLM_TEMPERATURE", "0.0")),
      default_max_tokens=default_max_tokens,
      max_model_len=int(max_model_len_env) if max_model_len_env else (
          128000 if provider == "zhipu" else None
      ),
      provider=provider,
    )

  @property
  def is_zhipu(self) -> bool:
    return _detect_provider(self.base_url, self.provider) == "zhipu"


@dataclass
class ChatResult:
  text: str
  content: Optional[str]
  reasoning: Optional[str]
  finish_reason: str
  model: str
  usage: Dict[str, int]
  raw: Dict[str, Any]


def _extract_text(
  message: Dict[str, Any],
  mode: OutputMode,
) -> tuple[str, Optional[str], Optional[str]]:
  content = message.get("content")
  reasoning = message.get("reasoning")

  content_s = content if isinstance(content, str) and content else None
  reasoning_s = reasoning if isinstance(reasoning, str) and reasoning else None

  if mode == OutputMode.CONTENT:
    text = content_s or ""
  elif mode == OutputMode.REASONING:
    text = reasoning_s or ""
  else:
    text = content_s or reasoning_s or ""

  return text, content_s, reasoning_s


class VLLMClient:
  """Thin wrapper around OpenAI-compatible HTTP API (vLLM or Zhipu)."""

  def __init__(self, config: Optional[VLLMConfig] = None):
    self.config = config or VLLMConfig.from_env()
    # Keep provider field consistent with base_url heuristics.
    self.config.provider = _detect_provider(
      self.config.base_url, self.config.provider
    )
    self._resolved_model: Optional[str] = None
    self._model_info: Dict[str, Dict[str, Any]] = {}

  @property
  def chat_path(self) -> str:
    # Zhipu: POST {base}/chat/completions
    # vLLM:  POST {base}/v1/chat/completions
    return "/chat/completions" if self.config.is_zhipu else "/v1/chat/completions"

  @property
  def base_url(self) -> str:
    return self.config.base_url.rstrip("/")

  def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
    """Return /v1/models entry for a model (cached)."""
    model_id = model or self.resolve_model()
    if model_id in self._model_info:
      return self._model_info[model_id]

    data = self._request("GET", "/v1/models")
    models = data.get("data") or []
    for entry in models:
      if isinstance(entry, dict) and entry.get("id"):
        self._model_info[str(entry["id"])] = entry

    if model_id not in self._model_info:
      raise VLLMError(f"Model not found on server: {model_id}")
    return self._model_info[model_id]

  def get_max_model_len(self, model: Optional[str] = None) -> int:
    if self.config.max_model_len is not None:
      return self.config.max_model_len
    info = self.get_model_info(model)
    max_len = info.get("max_model_len")
    if isinstance(max_len, int) and max_len > 0:
      return max_len
    return 32768

  def _request(
    self,
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    url = f"{self.base_url}{path}"
    data = None
    headers = {"Content-Type": "application/json"}
    if self.config.api_key:
      headers["Authorization"] = f"Bearer {self.config.api_key}"

    if body is not None:
      data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
      with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
        payload = resp.read().decode("utf-8")
        status = getattr(resp, "status", 200)
    except urllib.error.HTTPError as exc:
      detail = exc.read().decode("utf-8", errors="replace")
      raise VLLMError(f"HTTP {exc.code} {method} {path}: {detail}") from exc
    except urllib.error.URLError as exc:
      raise VLLMError(f"Failed to reach vLLM at {url}: {exc}") from exc

    if not payload.strip():
      return {"status": status}

    try:
      return json.loads(payload)
    except json.JSONDecodeError as exc:
      raise VLLMError(f"Invalid JSON from {url}: {payload[:500]}") from exc

  def health(self) -> bool:
    try:
      if self.config.is_zhipu:
        # Zhipu has no /health; a tiny chat proves auth + route.
        _ = self.ask("ping", max_tokens=8, temperature=0.0)
        return True
      url = f"{self.base_url}/health"
      req = urllib.request.Request(url, method="GET")
      with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
        return 200 <= getattr(resp, "status", 200) < 300
    except (urllib.error.URLError, urllib.error.HTTPError, VLLMError):
      return False

  def list_models(self) -> List[str]:
    if self.config.is_zhipu:
      # OpenAI-compat models list may be limited; prefer configured model.
      return [self.config.model or ZHIPU_DEFAULT_MODEL]
    data = self._request("GET", "/v1/models")
    models = data.get("data") or []
    return [m["id"] for m in models if isinstance(m, dict) and m.get("id")]

  def resolve_model(self) -> str:
    if self.config.model:
      return self.config.model
    if self._resolved_model:
      return self._resolved_model
    if self.config.is_zhipu:
      self._resolved_model = ZHIPU_DEFAULT_MODEL
      return self._resolved_model

    models = self.list_models()
    if not models:
      raise VLLMError("No models returned from /v1/models")
    self._resolved_model = models[0]
    return self._resolved_model

  @staticmethod
  def _resolve_max_tokens(
    max_tokens: Optional[int],
    default_max_tokens: Optional[int],
  ) -> Optional[int]:
    """Return explicit cap, or None to let vLLM use the full remaining context."""
    if max_tokens is not None:
      return max_tokens
    return default_max_tokens

  def chat(
    self,
    messages: Sequence[Message],
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
    output_mode: Optional[OutputMode] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
  ) -> ChatResult:
    """Call POST /v1/chat/completions and return structured output."""
    thinking = (
      self.config.enable_thinking if enable_thinking is None else enable_thinking
    )
    mode = output_mode or self.config.output_mode
    resolved_max_tokens = self._resolve_max_tokens(
      max_tokens, self.config.default_max_tokens
    )

    body: Dict[str, Any] = {
      "model": model or self.resolve_model(),
      "messages": list(messages),
      "temperature": (
        self.config.default_temperature if temperature is None else temperature
      ),
    }
    if self.config.is_zhipu:
      # glm-5.3-flash: thinking.type must be "enabled"; effort via reasoning_effort.
      model_l = (model or self.resolve_model() or "").lower()
      always_think = any(
        x in model_l for x in ("glm-5.3-flash", "glm-5.3-flashx", "glm-5.3")
      )
      if thinking or always_think:
        body["thinking"] = {"type": "enabled"}
        effort = (os.getenv("VLLM_REASONING_EFFORT") or "").strip().lower()
        if effort not in {"low", "high", "max"}:
          # Prefer low for free-pack extract/fill unless caller opts into deep think.
          effort = "high" if thinking else "low"
        body["reasoning_effort"] = effort
      else:
        body["thinking"] = {"type": "disabled"}
    else:
      body["chat_template_kwargs"] = {"enable_thinking": thinking}
    if resolved_max_tokens is not None:
      body["max_tokens"] = resolved_max_tokens
    body.update(self.config.extra_body)
    if extra_body:
      body.update(extra_body)
    body.update(kwargs)

    data = self._request("POST", self.chat_path, body)
    choices = data.get("choices") or []
    if not choices:
      raise VLLMError(f"Empty choices in response: {data}")

    choice = choices[0]
    message = choice.get("message") or {}
    text, content, reasoning = _extract_text(message, mode)
    usage = data.get("usage") or {}

    return ChatResult(
      text=text,
      content=content,
      reasoning=reasoning,
      finish_reason=str(choice.get("finish_reason", "")),
      model=str(data.get("model", body["model"])),
      usage={k: int(v) for k, v in usage.items() if isinstance(v, (int, float))},
      raw=data,
    )

  def ask(
    self,
    prompt: str,
    *,
    system: Optional[str] = None,
    **kwargs: Any,
  ) -> str:
    """Single-turn helper; returns generated text only."""
    messages: List[Message] = []
    if system:
      messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return self.chat(messages, **kwargs).text

  def ask_many(
    self,
    prompts: Sequence[str],
    *,
    system: Optional[str] = None,
    concurrency: int = 48,
    semaphore: Optional[threading.Semaphore] = None,
    **kwargs: Any,
  ) -> List[str]:
    """Fire many single-turn requests concurrently; preserve input order.

    vLLM continuous-batches in-flight requests, so client-side concurrency is
    the practical way to batch different prompts over the OpenAI chat API.

    Optional ``semaphore`` caps *global* in-flight asks (e.g. shared across
    multiple parallel Cleaner jobs hitting the same remote vLLM).
    """
    if not prompts:
      return []
    if concurrency < 1:
      raise ValueError("concurrency must be >= 1")

    def _ask_one(prompt: str) -> str:
      if semaphore is None:
        return self.ask(prompt, system=system, **kwargs)
      semaphore.acquire()
      try:
        return self.ask(prompt, system=system, **kwargs)
      finally:
        semaphore.release()

    if len(prompts) == 1 or concurrency == 1:
      return [_ask_one(p) for p in prompts]

    results: List[Optional[str]] = [None] * len(prompts)

    def _one(idx: int, prompt: str) -> tuple[int, str]:
      return idx, _ask_one(prompt)

    workers = min(concurrency, len(prompts))
    with ThreadPoolExecutor(max_workers=workers) as pool:
      futures = [pool.submit(_one, i, p) for i, p in enumerate(prompts)]
      for fut in as_completed(futures):
        idx, text = fut.result()
        results[idx] = text
    return [text if text is not None else "" for text in results]

  def complete(
    self,
    prompt: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
  ) -> str:
    """Call POST /v1/completions (non-chat) and return generated text."""
    resolved_max_tokens = self._resolve_max_tokens(
      max_tokens, self.config.default_max_tokens
    )

    body: Dict[str, Any] = {
      "model": model or self.resolve_model(),
      "prompt": prompt,
      "temperature": (
        self.config.default_temperature if temperature is None else temperature
      ),
    }
    if resolved_max_tokens is not None:
      body["max_tokens"] = resolved_max_tokens
    body.update(self.config.extra_body)
    body.update(kwargs)

    data = self._request("POST", "/v1/completions", body)
    choices = data.get("choices") or []
    if not choices:
      raise VLLMError(f"Empty choices in completion response: {data}")
    text = choices[0].get("text", "")
    return text if isinstance(text, str) else str(text)


def get_client(
  base_url: Optional[str] = None,
  model: Optional[str] = None,
  enable_thinking: Optional[bool] = None,
  **kwargs: Any,
) -> VLLMClient:
  """Factory for quick reuse in notebooks and scripts."""
  config = VLLMConfig.from_env()
  if base_url is not None:
    config.base_url = base_url.rstrip("/")
  if model is not None:
    config.model = model
  if enable_thinking is not None:
    config.enable_thinking = enable_thinking
  for key, value in kwargs.items():
    if hasattr(config, key):
      setattr(config, key, value)
  return VLLMClient(config)
