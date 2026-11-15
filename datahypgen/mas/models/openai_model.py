"""
OpenAI Model Adapter for DataGenHyp

Implements the BaseModel interface for OpenAI models.
"""

import json
import logging
import os
import re
from urllib.parse import parse_qs, urlparse
from typing import Dict, List, Optional, Any, Union
import httpx

import openai
from openai import AsyncOpenAI

try:
    from openai import AsyncAzureOpenAI
except ImportError:
    AsyncAzureOpenAI = None

from .base_model import BaseModel
from .structured_output import enforce_schema_with_retry, parse_json_object

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def _is_clawkey_endpoint(value: Optional[str]) -> bool:
    if not value or not isinstance(value, str):
        return False
    return "api.clawkey.vip" in value.lower()


def _is_deepseek_endpoint(value: Optional[str]) -> bool:
    if not value or not isinstance(value, str):
        return False
    return "api.deepseek.com" in value.lower()


def _resolve_env_like_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    match = _ENV_VAR_PATTERN.match(value.strip())
    if not match:
        return value

    env_name = match.group(1)
    resolved = os.environ.get(env_name)
    return resolved if resolved else None


def _looks_like_azure_endpoint(value: Optional[str]) -> bool:
    if not value or not isinstance(value, str):
        return False
    return ".azure.com" in value.lower()


def _normalize_azure_endpoint(raw_base_url: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Normalize Azure OpenAI endpoint inputs.

    Returns:
        (azure_endpoint, azure_deployment, api_version)
    Supported inputs:
    - https://<host>.openai.azure.com
    - https://<host>.openai.azure.com/v1
    - https://<host>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=...
    """
    if not raw_base_url:
        return None, None, None

    parsed = urlparse(raw_base_url)
    if not parsed.scheme or not parsed.netloc:
        return None, None, None

    azure_endpoint = f"{parsed.scheme}://{parsed.netloc}"
    api_version = parse_qs(parsed.query).get("api-version", [None])[0]
    deployment = None

    path = parsed.path.strip("/")
    if path:
        parts = path.split("/")
        if "deployments" in parts:
            idx = parts.index("deployments")
            if idx + 1 < len(parts):
                deployment = parts[idx + 1]

    return azure_endpoint, deployment, api_version


def _resolve_openai_base_url(explicit_base_url: Optional[str]) -> Optional[str]:
    return (
        _resolve_env_like_value(explicit_base_url)
        or os.environ.get("OPENAI_API_BASE_URL")
        or os.environ.get("BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
    )


def _strip_reasoning_content(text: str) -> str:
    if not isinstance(text, str):
        return text
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


def _normalize_wire_api(value: Optional[str]) -> str:
    if not value or not isinstance(value, str):
        return "chat_completions"

    normalized = value.strip().lower().replace(".", "_").replace("-", "_")
    if normalized in {"responses", "response"}:
        return "responses"
    return "chat_completions"


def _drop_none_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _normalize_prompt_cache_options(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("prompt_cache_options must be a dictionary when provided")
    options = _drop_none_values(dict(value))
    return options or None


def _extract_text_from_content_value(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return _strip_reasoning_content(content)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
                continue
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text_parts.append(str(getattr(item, "text", "")))
        if text_parts:
            return _strip_reasoning_content("".join(text_parts))
    return None


def _preview_text(value: Any, limit: int = 200) -> str:
    if value is None:
        return "None"
    text = str(value).replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def _get_value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _extract_text_from_response_output_content(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return _strip_reasoning_content(content)
    if not isinstance(content, list):
        return None

    text_parts: List[str] = []
    for item in content:
        item_type = _get_value(item, "type")
        if item_type not in {"output_text", "text"}:
            continue
        text_value = _get_value(item, "text")
        if isinstance(text_value, str) and text_value:
            text_parts.append(text_value)

    if text_parts:
        return _strip_reasoning_content("".join(text_parts))
    return None


def _extract_text_from_responses_output(output: Any) -> Optional[str]:
    if not isinstance(output, list):
        return None

    text_parts: List[str] = []
    for item in output:
        item_type = _get_value(item, "type")
        if item_type == "message":
            content_text = _extract_text_from_response_output_content(_get_value(item, "content"))
            if content_text:
                text_parts.append(content_text)
            continue

        if item_type in {"output_text", "text"}:
            text_value = _get_value(item, "text")
            if isinstance(text_value, str) and text_value:
                text_parts.append(text_value)

    if text_parts:
        return _strip_reasoning_content("".join(text_parts))
    return None


def _summarize_completion_response(response: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "response_type": type(response).__name__,
    }

    if isinstance(response, str):
        summary["text_preview"] = _preview_text(response)
        return summary

    if isinstance(response, dict):
        summary["keys"] = sorted(response.keys())
        if isinstance(response.get("output_text"), str):
            summary["output_text_preview"] = _preview_text(response.get("output_text"))
        choices = response.get("choices")
        summary["choices_count"] = len(choices) if isinstance(choices, list) else None
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                summary["first_choice_keys"] = sorted(first_choice.keys())
                summary["finish_reason"] = first_choice.get("finish_reason")
                message = first_choice.get("message")
                if isinstance(message, dict):
                    summary["message_keys"] = sorted(message.keys())
                    summary["content_preview"] = _preview_text(message.get("content"))
                    summary["reasoning_content_preview"] = _preview_text(message.get("reasoning_content"))
        return summary

    choices = getattr(response, "choices", None)
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        summary["output_text_preview"] = _preview_text(output_text)
    summary["choices_count"] = len(choices) if choices is not None else None
    if choices:
        first_choice = choices[0]
        summary["finish_reason"] = getattr(first_choice, "finish_reason", None)
        message = getattr(first_choice, "message", None)
        if message is not None:
            summary["message_type"] = type(message).__name__
            summary["content_type"] = type(getattr(message, "content", None)).__name__
            summary["content_preview"] = _preview_text(getattr(message, "content", None))
            summary["reasoning_content_preview"] = _preview_text(getattr(message, "reasoning_content", None))
            summary["refusal_preview"] = _preview_text(getattr(message, "refusal", None))
    return summary


def _extract_text_from_sse_payload(raw_text: str) -> Optional[str]:
    if not isinstance(raw_text, str) or not re.search(r"^\s*data:", raw_text, re.MULTILINE):
        return None

    text_parts: List[str] = []
    # Responses API 事件流：网关对非流式 /responses 请求也按 SSE 返回。终态帧优先级
    # completed.response.output > output_text.done.text > 拼接 output_text.delta。
    responses_completed: Optional[str] = None
    responses_done_parts: List[str] = []
    responses_delta_parts: List[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue

        payload = stripped[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")
        if isinstance(event_type, str) and event_type.startswith("response."):
            if event_type == "response.completed":
                completed_text = _extract_text_from_responses_output(
                    _get_value(event.get("response"), "output")
                )
                if completed_text:
                    responses_completed = completed_text
            elif event_type == "response.output_text.done":
                text_value = event.get("text")
                if isinstance(text_value, str) and text_value:
                    responses_done_parts.append(text_value)
            elif event_type == "response.output_text.delta":
                delta_value = event.get("delta")
                if isinstance(delta_value, str):
                    responses_delta_parts.append(delta_value)
            continue

        choices = event.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta_text = _extract_text_from_content_value(choice.get("delta", {}).get("content"))
                if delta_text:
                    text_parts.append(delta_text)
                    continue
                message_text = _extract_text_from_content_value(choice.get("message", {}).get("content"))
                if message_text:
                    text_parts.append(message_text)
                    continue
                text_text = _extract_text_from_content_value(choice.get("text"))
                if text_text:
                    text_parts.append(text_text)

        event_content = _extract_text_from_content_value(event.get("content"))
        if event_content:
            text_parts.append(event_content)

    if responses_completed:
        return responses_completed
    if responses_done_parts:
        return _strip_reasoning_content("".join(responses_done_parts))
    if responses_delta_parts:
        return _strip_reasoning_content("".join(responses_delta_parts))
    if text_parts:
        return "".join(text_parts)
    return raw_text.strip()


def _extract_text_from_completion_response(response: Any) -> str:
    """
    兼容标准 OpenAI SDK 对象以及部分 OpenAI-compatible 端点返回的简化 dict / str。
    """
    if isinstance(response, str):
        sse_text = _extract_text_from_sse_payload(response)
        if sse_text is not None:
            return sse_text
        return _strip_reasoning_content(response)

    if isinstance(response, dict):
        output_text = response.get("output_text")
        if isinstance(output_text, str):
            return _strip_reasoning_content(output_text)
        output_content_text = _extract_text_from_responses_output(response.get("output"))
        if output_content_text is not None:
            return output_content_text
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                content_text = _extract_text_from_content_value(message.get("content"))
                if content_text is not None:
                    return content_text
        if "content" in response and isinstance(response["content"], str):
            return _strip_reasoning_content(response["content"])
        return json.dumps(response, ensure_ascii=False)

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return _strip_reasoning_content(output_text)

    output_content_text = _extract_text_from_responses_output(getattr(response, "output", None))
    if output_content_text is not None:
        return output_content_text

    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        if message is not None:
            content_text = _extract_text_from_content_value(getattr(message, "content", None))
            if content_text is not None:
                return content_text

    raise TypeError(f"Unsupported completion response type: {type(response).__name__}")


class OpenAIModel(BaseModel):
    """OpenAI implementation of the BaseModel interface."""
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                base_url: Optional[str] = None,
                model_name: str = "gpt-5.4-mini",
                wire_api: Optional[str] = None,
                use_native_json_schema: bool = True,
                prompt_cache_key: Optional[str] = None,
                prompt_cache_options: Optional[Dict[str, Any]] = None,
                prompt_cache_retention: Optional[str] = None,
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 1800):
        """
        Initialize the OpenAI model adapter.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model_name: Model identifier to use (e.g., "gpt-5.4-mini")
            wire_api: Transport API to use ("chat_completions" or "responses")
            prompt_cache_key: Stable key for OpenAI prompt-cache routing
            prompt_cache_options: OpenAI prompt cache options, e.g. {"ttl": "30m"}
            prompt_cache_retention: Prompt-cache retention policy ("in-memory" or "24h")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
        """
        self.api_key = _resolve_env_like_value(api_key) or os.environ.get("OPENAI_API_KEY")
        self.base_url = _resolve_openai_base_url(base_url)
        self.api_version = (
            os.environ.get("AZURE_OPENAI_API_VERSION")
            or os.environ.get("OPENAI_API_VERSION")
        )
        self.azure_deployment = None
        self.is_azure = _looks_like_azure_endpoint(self.base_url)
        self.is_clawkey = _is_clawkey_endpoint(self.base_url)
        self.is_deepseek = _is_deepseek_endpoint(self.base_url)
        self.wire_api = _normalize_wire_api(wire_api)
        self.use_native_json_schema = bool(use_native_json_schema)
        self.prompt_cache_key = _resolve_env_like_value(prompt_cache_key)
        self.prompt_cache_options = _normalize_prompt_cache_options(prompt_cache_options)
        self.prompt_cache_retention = _resolve_env_like_value(prompt_cache_retention)
        if self.prompt_cache_retention not in {None, "in-memory", "24h"}:
            raise ValueError(
                "prompt_cache_retention must be omitted, 'in-memory', or '24h'"
            )
        self._chat_capabilities: Optional[Dict[str, bool]] = None
        if not self.api_key:
            logger.warning("OpenAI API key not provided. Please set OPENAI_API_KEY environment variable.")
            
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        # 使用 httpx.Timeout 对象明确设置所有超时维度，防止 API 挂起
        # connect: 建立连接超时；read: 每次读取数据块超时；write: 写入请求体超时；pool: 连接池超时
        # read 设置为配置的 timeout，确保整体响应时间有上限；write 和 connect 给足够余量处理大内容
        timeout_config = httpx.Timeout(
            connect=60.0,      # 60 秒连接超时，处理网络波动
            read=float(timeout),  # 使用配置的 timeout 作为每次读取超时
            write=120.0,       # 120 秒写入超时，处理长 prompt 上传
            pool=10.0          # 10 秒连接池超时
        )
        # 显式配置连接池限制，避免连接泄漏或池耗尽
        limits = httpx.Limits(
            max_connections=100,        # 最大总连接数
            max_keepalive_connections=20,  # 最大保持活动连接数
            keepalive_expiry=30.0       # 空闲连接 30 秒后过期
        )
        self.http_client = httpx.AsyncClient(
            trust_env=True,
            timeout=timeout_config,
            limits=limits
        )
        
        try:
            if self.is_azure:
                if AsyncAzureOpenAI is None:
                    raise TypeError("openai package does not provide AsyncAzureOpenAI")

                azure_endpoint, azure_deployment, parsed_api_version = _normalize_azure_endpoint(self.base_url)
                self.api_version = parsed_api_version or self.api_version
                self.azure_deployment = azure_deployment or self.model_name

                if not azure_endpoint:
                    raise ValueError(f"Invalid Azure OpenAI endpoint: {self.base_url}")
                if not self.api_version:
                    raise ValueError(
                        "Azure OpenAI requires api_version. Set OPENAI_API_VERSION or AZURE_OPENAI_API_VERSION."
                    )

                # http_client 已经配置了完整的 timeout，不应该再传递 timeout 参数
                self.client = AsyncAzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=azure_endpoint,
                    api_version=self.api_version,
                    http_client=self.http_client,
                )
                logger.info(
                    "Azure OpenAI client initialized with model: %s deployment: %s endpoint: %s api_version: %s",
                    self.model_name,
                    self.azure_deployment,
                    azure_endpoint,
                    self.api_version,
                )
            else:
                # http_client 已经配置了完整的 timeout，不应该再传递 timeout 参数
                # 同时传递会导致冲突，SDK 可能用传入的 timeout 覆盖 http_client 的配置
                client_kwargs = {
                    "api_key": self.api_key,
                    "http_client": self.http_client,
                    # Retry policy is handled by BaseAgent so provider-unavailable
                    # responses use the configured one-minute retry interval.
                    "max_retries": 0,
                }
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url

                self.client = AsyncOpenAI(**client_kwargs)
                logger.info(
                    "OpenAI client initialized with model: %s via %s (wire_api=%s)",
                    self.model_name,
                    self.base_url or "provider_default",
                    self.wire_api,
                )
        except TypeError as e:
            logger.warning(f"Error initializing OpenAI client: {e}")
            self.client = None

    def _apply_prompt_cache_request_options(self, request_kwargs: Dict[str, Any]) -> None:
        if self.prompt_cache_key:
            request_kwargs["prompt_cache_key"] = self.prompt_cache_key
        if self.prompt_cache_retention:
            request_kwargs["prompt_cache_retention"] = self.prompt_cache_retention
        if not self.prompt_cache_options:
            return

        extra_body = request_kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            extra_body = dict(extra_body)
        elif extra_body is None:
            extra_body = {}
        else:
            raise ValueError(
                "extra_body must be a dictionary when prompt_cache_options is configured"
            )
        # The installed OpenAI SDK exposes prompt_cache_key/retention directly,
        # but not prompt_cache_options yet. extra_body keeps GPT-5.6 cache
        # controls available without pinning a newer SDK.
        extra_body["prompt_cache_options"] = self.prompt_cache_options
        request_kwargs["extra_body"] = extra_body

    async def _probe_chat_capabilities(self) -> Dict[str, bool]:
        if self._chat_capabilities is not None:
            return self._chat_capabilities

        default_caps = {
            "supports_messages_only": True,
            "supports_temperature": True,
            "supports_max_tokens": True,
            "supports_response_format": True,
        }
        if not self.is_clawkey or self.client is None:
            self._chat_capabilities = default_caps
            return self._chat_capabilities

        probe_messages = [{"role": "user", "content": "Reply with OK only."}]
        model_name = self.azure_deployment if self.is_azure else self.model_name

        async def _try_probe(**probe_kwargs) -> bool:
            try:
                await self.client.chat.completions.create(
                    model=model_name,
                    messages=probe_messages,
                    **probe_kwargs,
                )
                return True
            except Exception as exc:
                logger.warning(
                    "[clawkey probe] model=%s base=%s kwargs=%s rejected: %s",
                    self.model_name,
                    self.base_url,
                    sorted(probe_kwargs.keys()),
                    exc,
                )
                return False

        supports_messages_only = await _try_probe()
        capabilities = {
            "supports_messages_only": supports_messages_only,
            "supports_temperature": False,
            "supports_max_tokens": False,
            "supports_response_format": False,
        }
        if supports_messages_only:
            capabilities["supports_temperature"] = await _try_probe(temperature=0.1)
            capabilities["supports_max_tokens"] = await _try_probe(max_tokens=16)
            capabilities["supports_response_format"] = await _try_probe(
                response_format={"type": "json_object"}
            )

        self._chat_capabilities = capabilities
        logger.info(
            "[clawkey probe] model=%s capabilities=%s",
            self.model_name,
            capabilities,
        )
        return self._chat_capabilities

    def _build_responses_request(
        self,
        *,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        request_kwargs = _drop_none_values(dict(kwargs))
        input_messages: List[Dict[str, Any]] = []
        instruction_parts: List[str] = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                content = "".join(text_parts)
            if content is None:
                content = ""
            content_text = str(content)

            if role == "system":
                if content_text:
                    instruction_parts.append(content_text)
                continue

            input_messages.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": content_text}],
                }
            )

        if not input_messages:
            input_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": ""}],
                }
            )

        request_kwargs["model"] = self.model_name
        request_kwargs["input"] = input_messages
        if instruction_parts:
            request_kwargs["instructions"] = "\n\n".join(instruction_parts)
        self._apply_prompt_cache_request_options(request_kwargs)
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if max_tokens is not None:
            request_kwargs["max_output_tokens"] = max_tokens
        if stop_sequences is not None:
            request_kwargs["stop"] = stop_sequences
        if json_schema is not None and self.use_native_json_schema:
            request_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "datahypgen_structured_output",
                    "schema": json_schema,
                    "strict": True,
                }
            }

        return request_kwargs

    async def _create_responses_completion(
        self,
        *,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        request_kwargs = self._build_responses_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            json_schema=json_schema,
            **kwargs,
        )
        return await self.client.responses.create(**request_kwargs)

    async def _create_chat_completion(
        self,
        *,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        request_kwargs = _drop_none_values(dict(kwargs))
        self._apply_prompt_cache_request_options(request_kwargs)
        if self.is_deepseek and self.model_name.startswith("deepseek-v4"):
            extra_body = request_kwargs.get("extra_body")
            if isinstance(extra_body, dict):
                extra_body = dict(extra_body)
            else:
                extra_body = {}
            thinking = extra_body.get("thinking")
            if not isinstance(thinking, dict) or thinking.get("type") != "disabled":
                extra_body["thinking"] = {"type": "disabled"}
                request_kwargs["extra_body"] = extra_body
                logger.info(
                    "Applying DeepSeek thinking=disabled for model=%s base=%s",
                    self.model_name,
                    self.base_url,
                )
        if self.is_clawkey:
            capabilities = await self._probe_chat_capabilities()
            if not capabilities.get("supports_messages_only", False):
                raise ValueError(
                    f"Clawkey chat.completions probe failed for model={self.model_name}; "
                    "messages-only request was rejected."
                )
            if temperature is not None:
                if capabilities.get("supports_temperature", False):
                    request_kwargs["temperature"] = temperature
                else:
                    logger.info("[clawkey probe] dropping unsupported temperature for model=%s", self.model_name)
            if max_tokens is not None:
                if capabilities.get("supports_max_tokens", False):
                    request_kwargs["max_tokens"] = max_tokens
                else:
                    logger.info("[clawkey probe] dropping unsupported max_tokens for model=%s", self.model_name)
            if stop_sequences is not None:
                request_kwargs["stop"] = stop_sequences
            if (
                "response_format" in request_kwargs
                and not capabilities.get("supports_response_format", False)
            ):
                logger.info("[clawkey probe] dropping unsupported response_format for model=%s", self.model_name)
                request_kwargs.pop("response_format", None)
        else:
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            if stop_sequences is not None:
                request_kwargs["stop"] = stop_sequences

        try:
            return await self.client.chat.completions.create(
                model=self.azure_deployment if self.is_azure else self.model_name,
                messages=messages,
                **request_kwargs,
            )
        except openai.BadRequestError as e:
            if self.is_clawkey:
                raise
            # 部分 OpenAI-compatible reasoning 端点会拒绝 max_tokens，但在去掉该字段后可正常返回。
            if (
                max_tokens is not None
                and not self.is_azure
                and self.base_url
                and "api.clawkey.vip" in self.base_url
            ):
                logger.warning(
                    "Retrying chat completion without max_tokens for provider %s model %s due to BadRequestError: %s",
                    self.base_url,
                    self.model_name,
                    e,
                )
                retry_kwargs = dict(kwargs)
                if temperature is not None:
                    retry_kwargs["temperature"] = temperature
                if stop_sequences is not None:
                    retry_kwargs["stop"] = stop_sequences
                return await self.client.chat.completions.create(
                    model=self.azure_deployment if self.is_azure else self.model_name,
                    messages=messages,
                    **retry_kwargs,
                )
            raise

    async def _create_completion(
        self,
        *,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        if self.wire_api == "responses":
            return await self._create_responses_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                json_schema=json_schema,
                **kwargs,
            )

        return await self._create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            **kwargs,
        )
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using OpenAI API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """

        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = await self._create_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
                **kwargs
            )
            result_text = _extract_text_from_completion_response(response)
            if not result_text.strip():
                logger.error(
                    "OpenAI-compatible model returned empty text response. model=%s base=%s summary=%s",
                    self.model_name,
                    self.base_url,
                    _summarize_completion_response(response),
                )
            return result_text
        except Exception as e:
            # 提供更详细的错误信息用于诊断
            error_msg = str(e)
            logger.error(f"Error generating response from OpenAI: {error_msg}")
            logger.error(f"  Model: {self.model_name}")
            logger.error(f"  Base URL: {self.base_url}")
            logger.error(f"  Is Azure: {self.is_azure}")
            logger.error(f"  Azure deployment: {self.azure_deployment}")
            logger.error(f"  Wire API: {self.wire_api}")
            logger.error(f"  API version: {self.api_version}")
            logger.error(f"  API Key present: {bool(self.api_key)}")
            if hasattr(e, 'status_code'):
                logger.error(f"  HTTP Status: {e.status_code}")
            if hasattr(e, 'response'):
                logger.error(f"  Response: {e.response}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str,
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """

        required_fields = json_schema.get("required", []) if isinstance(json_schema, dict) else []
        enum_guidance: List[str] = []

        def collect_enum_paths(node: Any, path: str = "$") -> None:
            if not isinstance(node, dict):
                return
            enum_values = node.get("enum")
            if isinstance(enum_values, list) and enum_values:
                enum_guidance.append(f"{path} must be exactly one of {enum_values}")
            properties = node.get("properties")
            if isinstance(properties, dict):
                for key, child in properties.items():
                    collect_enum_paths(child, f"{path}.{key}")
            items = node.get("items")
            if isinstance(items, dict):
                collect_enum_paths(items, f"{path}[*]")

        collect_enum_paths(json_schema)
        enum_instruction = (
            "\nEnum constraints are strict; never invent continuous/intermediate values.\n"
            + "\n".join(enum_guidance)
            if enum_guidance
            else ""
        )
        schema_instruction = (
            "Return only one complete valid JSON object matching this JSON schema. "
            "Every field listed in a `required` array must be present. Do not add "
            "undeclared fields or Markdown commentary.\n"
            "Exact required top-level keys: "
            + ", ".join(str(item) for item in required_fields)
            + enum_instruction
            + f"\nJSON schema:\n{json.dumps(json_schema, ensure_ascii=False)}"
        )
        enhanced_system_prompt = (
            f"{system_prompt}\n\n{schema_instruction}" if system_prompt else schema_instruction
        )

        async def request_json(current_prompt: str) -> Dict[str, Any]:
            response = await self._create_completion(
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": current_prompt},
                ],
                temperature=temperature if temperature is not None else self.temperature,
                json_schema=json_schema if self.wire_api == "responses" else None,
                response_format={"type": "json_object"} if self.wire_api != "responses" else None,
                **kwargs,
            )
            result_text = _extract_text_from_completion_response(response)
            if not result_text.strip():
                raise ValueError("Model returned empty response while JSON output was expected")
            return parse_json_object(result_text, "OpenAI")

        try:
            result = await request_json(prompt)

            async def retry(violations: List[str]) -> Dict[str, Any]:
                correction_prompt = (
                    f"{prompt}\n\n---\n"
                    "The previous JSON violated the required schema:\n"
                    + "\n".join(f"- {violation}" for violation in violations)
                    + "\n\nReturn one complete corrected JSON object from scratch. "
                    "Include every required field, do not add undeclared fields, and do not explain."
                    + (" Use only the exact enum values listed in the schema; do not output percentages or arbitrary scores." if enum_guidance else "")
                )
                return await request_json(correction_prompt)

            return await enforce_schema_with_retry(
                payload=result,
                schema=json_schema,
                retry=retry,
                provider_name="OpenAI",
                max_correction_retries=2,
            )
        except Exception as e:
            logger.error(f"Error generating JSON response from OpenAI: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            response = await self.client.embeddings.create(
                model=self.azure_deployment if self.is_azure else "text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OpenAIModel':
        """
        Create an OpenAI model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Debug configuration dictionary with model settings
            
        Returns:
            Configured OpenAIModel instance
        """
        return cls(
            api_key=_resolve_env_like_value(config.get("api_key")),
            base_url=(
                _resolve_env_like_value(config.get("api_base"))
                or _resolve_env_like_value(config.get("base_url"))
            ),
            model_name=config.get("model_name", "gpt-3.5-turbo"),
            wire_api=config.get("wire_api"),
            use_native_json_schema=config.get("use_native_json_schema", True),
            prompt_cache_key=config.get("prompt_cache_key"),
            prompt_cache_options=config.get("prompt_cache_options"),
            prompt_cache_retention=config.get("prompt_cache_retention"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 1800)
        ) 
