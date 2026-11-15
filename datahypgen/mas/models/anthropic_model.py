"""Anthropic Messages API adapter for DataGenHyp."""

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Union

from .base_model import BaseModel, EmptyResponseError, ModelRefusalError
from .structured_output import enforce_schema_with_retry, parse_json_object

try:
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover - reported when the provider is selected
    AsyncAnthropic = None


logger = logging.getLogger(__name__)

def _extract_text(response: Any) -> str:
    """Extract text blocks from an Anthropic response or a relay-compatible dict."""
    content = response.get("content") if isinstance(response, dict) else getattr(response, "content", None)
    if not isinstance(content, list):
        return ""

    parts: List[str] = []
    for block in content:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if block_type != "text":
            continue
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _content_block_types(response: Any) -> List[str]:
    content = response.get("content") if isinstance(response, dict) else getattr(response, "content", None)
    if not isinstance(content, list):
        return []
    return [
        str(block.get("type") if isinstance(block, dict) else getattr(block, "type", None))
        for block in content
    ]


def _raw_response_repr(response: Any, limit: int = 3000) -> str:
    try:
        if hasattr(response, "model_dump"):
            payload = response.model_dump()
        else:
            payload = response
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        text = repr(response)
    return text if len(text) <= limit else text[:limit] + f"...(+{len(text) - limit} chars)"


def _stop_reason(response: Any) -> Optional[str]:
    return response.get("stop_reason") if isinstance(response, dict) else getattr(response, "stop_reason", None)


def _is_truncated(response: Any) -> bool:
    return _stop_reason(response) == "max_tokens"


def _raise_if_refused(response: Any) -> None:
    if _stop_reason(response) != "refusal":
        return
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    output_tokens = (
        usage.get("output_tokens") if isinstance(usage, dict) else getattr(usage, "output_tokens", None)
    ) if usage is not None else None
    # output_tokens<=1 是输入阶段被分类器拦截，重发同一请求结果不变；
    # 更大的值是生成中途被掐断，取决于本次生成内容，重试一次有意义。
    phase = "input" if isinstance(output_tokens, int) and output_tokens <= 1 else "generation"
    logger.warning(
        "Anthropic refusal (%s phase): output_tokens=%s content_block_types=%s",
        phase,
        output_tokens,
        _content_block_types(response),
    )
    raise ModelRefusalError(
        f"Anthropic safety classifier refused the request during {phase} "
        f"(stop_reason=refusal, output_tokens={output_tokens})"
    )


def _cached_system(text: str) -> List[Dict[str, Any]]:
    """Mark the system prompt as a prompt-cache breakpoint; it is identical across calls of one reviewer."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _log_usage(response: Any) -> None:
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return

    def field(name: str) -> int:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        return int(value) if isinstance(value, int) else 0

    logger.info(
        "Anthropic usage: input=%d cache_read=%d cache_create=%d output=%d",
        field("input_tokens"),
        field("cache_read_input_tokens"),
        field("cache_creation_input_tokens"),
        field("output_tokens"),
    )


_OUTPUT_BUDGET_HEADROOM = 4


class AnthropicModel(BaseModel):
    """Native Anthropic Messages API implementation of the model contract."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "claude-sonnet-4-5",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 1800,
        anthropic_version: str = "2023-06-01",
    ):
        super().__init__()
        if AsyncAnthropic is None:
            raise ImportError("anthropic package is required for the anthropic provider")

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("ANTHROPIC_BASE_URL")
            or os.environ.get("ANTHROPIC_API_BASE_URL")
        )
        # 规范化 base_url：移除末尾的 /v1，因为 Anthropic SDK 会自动添加
        if self.base_url and self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.anthropic_version = anthropic_version
        if not self.api_key:
            logger.warning("Anthropic API key not provided; set ANTHROPIC_API_KEY")

        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": timeout,
            "max_retries": 1,
            "default_headers": {"anthropic-version": anthropic_version},
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = AsyncAnthropic(**client_kwargs)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        request: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            request["system"] = _cached_system(system_prompt)
        if stop_sequences:
            request["stop_sequences"] = stop_sequences
        request.update(kwargs)

        response = await self._create_message(request)
        result = _extract_text(response)
        if not result.strip():
            raise EmptyResponseError("Anthropic returned an empty text response")
        return result

    async def _create_message(self, request: Dict[str, Any]) -> Any:
        # 非流式：中转网关的 SSE 转发会丢 content_block 事件，SDK 拼出 content=[] 但
        # usage 正常，整批评估调用全部空返回。非流式首字节等待长，但至少响应完整。
        response = await self.client.messages.create(**request)
        _log_usage(response)
        _raise_if_refused(response)
        return response

    async def generate_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        default: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # 不用强制 tool_use：项目所经的中转网关用 XML 模拟 tool_use，嵌套 schema 必然被改坏；
        # 纯文本 JSON + 本地 schema 校验是唯一稳定的路径。
        request_kwargs = dict(kwargs)
        target_max_tokens = request_kwargs.pop("max_tokens", self.max_tokens)
        # 配置的 max_tokens 只作为目标预算写进 prompt；API 上限留余量，避免完整对象被砍掉
        api_max_tokens = target_max_tokens * _OUTPUT_BUDGET_HEADROOM
        json_system = (
            (f"{system_prompt}\n\n" if system_prompt else "")
            + f"Keep the whole response within roughly {target_max_tokens} output tokens.\n"
            "Return only one complete valid JSON object matching this JSON schema. "
            "Every field listed in a `required` array must be present. Do not add "
            "undeclared fields, XML tags, or Markdown commentary. Respect every "
            "`maxLength`, `maxItems`, `enum` and item `type` exactly: where items are "
            "typed `string`, emit plain strings, not objects.\n"
            f"JSON schema:\n{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
        )
        request_kwargs.update(
            {
                "model": self.model_name,
                "max_tokens": api_max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
                "system": _cached_system(json_system),
            }
        )

        async def request_json(current_prompt: str) -> Dict[str, Any]:
            text = ""
            for attempt in range(2):
                # 中转网关按请求内容缓存响应，且会缓存 text="" 的坏响应；原样重发只会
                # 再次命中同一条缓存，所以第二次带 nonce 让请求体不同。
                effective_prompt = (
                    current_prompt if attempt == 0
                    else f"{current_prompt}\n\n<!-- request-nonce: {uuid.uuid4().hex} -->"
                )
                response = await self._create_message(
                    {**request_kwargs, "messages": [{"role": "user", "content": effective_prompt}]}
                )
                if _is_truncated(response):
                    raise ValueError(
                        f"Anthropic output was truncated at max_tokens={api_max_tokens} "
                        f"(target budget {target_max_tokens}); the JSON object is incomplete"
                    )
                text = _extract_text(response)
                if text.strip():
                    break
                logger.warning(
                    "Anthropic empty text diagnostics (attempt %d): stop_reason=%r raw=%s",
                    attempt + 1,
                    _stop_reason(response),
                    _raw_response_repr(response),
                )
            if not text.strip():
                raise EmptyResponseError(
                    "Anthropic returned an empty completion twice (second attempt with cache-busting nonce); "
                    f"content block types={_content_block_types(response)}"
                )
            return parse_json_object(text, "Anthropic")

        async def retry(violations: List[str]) -> Dict[str, Any]:
            return await request_json(
                f"{prompt}\n\n---\n"
                "The previous output violated the required schema:\n"
                + "\n".join(f"- {violation}" for violation in violations)
                + "\n\nReturn one complete corrected object. Do not add undeclared fields."
            )

        try:
            return await enforce_schema_with_retry(
                payload=await request_json(prompt),
                schema=schema,
                retry=retry,
                provider_name="Anthropic",
            )
        except Exception:
            if default is not None:
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API; configure a separate embedding provider"
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AnthropicModel":
        return cls(
            api_key=config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=(
                config.get("api_base")
                or config.get("base_url")
                or os.environ.get("ANTHROPIC_BASE_URL")
                or os.environ.get("ANTHROPIC_API_BASE_URL")
            ),
            model_name=config.get("model_name", "claude-sonnet-4-5"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 1800),
            anthropic_version=config.get("anthropic_version", "2023-06-01"),
        )
