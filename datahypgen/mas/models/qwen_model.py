"""
Qwen Model Adapter for DataGenHyp

基于 OpenAI 兼容协议的 Qwen模型适配器，实现 BaseModel 接口。
默认读取环境变量 QWEN_API_KEY 与QWEN_BASE_URL，也支持从配置中显式传入。
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Any, Union

import httpx
from openai import AsyncOpenAI
from json_repair import repair_json

try:  # jsonschema 缺失时退化为原有的"仅语法校验"行为，不改变调用契约。
    from jsonschema.validators import validator_for as _jsonschema_validator_for
except Exception:  # pragma: no cover - 依赖缺失时的降级路径
    _jsonschema_validator_for = None

from .base_model import BaseModel

logger = logging.getLogger(__name__)

# 单次纠错重试中回报给模型的最大违规条数，避免超长 prompt 反而挤掉输出预算。
_MAX_REPORTED_SCHEMA_VIOLATIONS = 8

_ENV_VAR_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


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


def _resolve_qwen_base_url(explicit_base_url: Optional[str]) -> Optional[str]:
    return (
        _resolve_env_like_value(explicit_base_url)
        or os.environ.get("QWEN_API_BASE_URL")
        or os.environ.get("OPENAI_API_BASE_URL")
        or os.environ.get("BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
    )


class QwenModel(BaseModel):
    """
    Qwen 模型实现，适配 OpenAI Chat Completions 接口。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "qwen3.5-plus",
        embedding_model_name: str = "text-embedding-v4",
        embedding_batch_size: int = 20,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        timeout: int = 60,
    ):
        """
        初始化 Qwen模型适配器。

        Args:
            api_key: API key（默认读取QWEN_API_KEY 环境变量）
            base_url: OpenAI 兼容服务地址（默认读取 QWEN_API_BASE_URL 环境变量）
            model_name: 模型名称
            embedding_model_name: embedding 模型名称
            embedding_batch_size: embedding 单次请求最大文本条数
            max_tokens: 默认最大输出 token 数
            temperature: 默认采样温度
            timeout: 请求超时时间（暂未显式使用，仅保留接口）
        """
        self.api_key = _resolve_env_like_value(api_key) or os.environ.get("QWEN_API_KEY")
        if not self.api_key:
            logger.warning("QWEN_API_KEY not provided. Please set QWEN_API_KEY env variable.")

        # 兼容 config 中的 api_base / base_url
        self.base_url = _resolve_qwen_base_url(base_url)
        if not self.base_url:
            logger.warning(
                "Qwen base URL not provided. Please set QWEN_API_BASE_URL or OPENAI_API_BASE_URL"
            )

        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.embedding_batch_size = embedding_batch_size if embedding_batch_size > 0 else 20
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        # 明确设置所有超时阶段，确保 read timeout 生效
        timeout_config = httpx.Timeout(
            connect=60.0,  # 连接超时 60 秒
            read=float(self.timeout),  # 读取超时使用配置值（默认 600 秒）
            write=60.0,  # 写入超时 60 秒
            pool=10.0,  # 连接池超时 10 秒
        )
        self.http_client = httpx.AsyncClient(trust_env=True, timeout=timeout_config)

        try:
            client_kwargs = {
                "api_key": self.api_key,
                "http_client": self.http_client,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self.client = AsyncOpenAI(**client_kwargs)
            logger.info(
                "Qwen client initialized with model: %s via %s",
                self.model_name,
                self.base_url or "provider_default",
            )
        except TypeError as e:
            logger.warning(f"Error initializing Qwen client: {e}")
            self.client = None

    def _normalize_json_text(self, text: str) -> str:
        normalized = (text or "").strip()
        if normalized.startswith("```"):
            normalized = re.sub(r"^```(?:json)?\s*", "", normalized, flags=re.IGNORECASE)
            normalized = re.sub(r"\s*```$", "", normalized)
        return normalized.strip()

    @staticmethod
    def _extract_completion_text(response: Any) -> str:
        response_text = response.choices[0].message.content or ""
        if "</think>" in response_text:
            return response_text.split("</think>", 1)[1].strip()
        return response_text

    def _parse_json_result_text(self, result_text: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(result_text)
        except Exception:
            preview = result_text[:500]
            logger.error(f"Model returned invalid JSON: {preview}")
            repaired = repair_json(result_text)
            if repaired:
                try:
                    parsed = json.loads(repaired)
                except Exception:
                    logger.error(f"Repaired JSON still invalid: {repaired[:500]}")
                    raise ValueError("Model did not return valid JSON after repair")
            else:
                raise ValueError("Model did not return valid JSON")

        if self._looks_like_schema_echo(parsed):
            logger.error("Qwen returned a JSON schema echo instead of a JSON instance.")
            raise ValueError("Model echoed the JSON schema instead of returning an instance")

        return parsed

    @staticmethod
    def _looks_like_truncated_json(text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False

        stack: List[str] = []
        in_string = False
        escaped = False
        for ch in stripped:
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch == "}":
                if not stack or stack[-1] != "{":
                    return False
                stack.pop()
            elif ch == "]":
                if not stack or stack[-1] != "[":
                    return False
                stack.pop()

        if in_string or stack:
            return True
        return stripped[-1] not in {"}", "]"}

    def _apply_thinking_disabled(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        request_kwargs = dict(kwargs)
        extra_body = request_kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            merged_extra_body = dict(extra_body)
        else:
            merged_extra_body = {}

        merged_extra_body["thinking"] = {"type": "disabled"}
        request_kwargs["extra_body"] = merged_extra_body
        return request_kwargs

    def _build_json_system_prompt(
        self,
        json_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> str:
        schema_text = json.dumps(json_schema, ensure_ascii=False)
        json_instruction = (
            "Return exactly one JSON object instance that satisfies the schema below.\n"
            "Do not output the schema itself.\n"
            "Do not explain.\n"
            "Do not use markdown fences.\n"
            f"JSON schema:\n{schema_text}"
        )
        if system_prompt:
            return f"{system_prompt}\n\n{json_instruction}"
        return json_instruction

    @staticmethod
    def _looks_like_schema_echo(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False

        schema_like_keys = {
            "$defs",
            "$schema",
            "additionalProperties",
            "allOf",
            "anyOf",
            "definitions",
            "description",
            "enum",
            "items",
            "oneOf",
            "properties",
            "required",
            "title",
            "type",
        }
        payload_keys = set(payload.keys())
        if not payload_keys:
            return False

        if "properties" in payload_keys and payload_keys.issubset(schema_like_keys):
            return True
        if "$defs" in payload_keys and payload_keys.issubset(schema_like_keys):
            return True
        return False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        生成文本回复。
        """
        if self.client is None:
            raise RuntimeError("Qwen client is not initialized.")

        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        request_kwargs = self._apply_thinking_disabled(kwargs)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **request_kwargs,
            )

            response_text = response.choices[0].message.content

            # 如果模型带有 <think></think>，仅返回思考后的答案部分
            if response_text and "</think>" in response_text:
                answer_text = response_text.split("</think>", 1)[1].strip()
            else:
                answer_text = response_text

            return answer_text
        except Exception as e:
            logger.error(f"Error generating response from QwenModel: {e}")
            raise

    @staticmethod
    def _iter_schema_errors(payload: Any, json_schema: Dict[str, Any]) -> List[Any]:
        """Return raw jsonschema errors, or [] when validation is unavailable.

        本适配器把 schema 作为 system prompt 的软提示下发（阿里云 compatible-mode
        对 qwen 的 response_format=json_schema 支持不确定），因此语法合法但字段缺失
        的响应此前会被静默放行。这里补上确定性校验；jsonschema 不可用或 schema 本身
        非法时返回空列表，保持既有行为。
        """
        if _jsonschema_validator_for is None or not isinstance(json_schema, dict):
            return []
        try:
            validator_cls = _jsonschema_validator_for(json_schema)
            validator_cls.check_schema(json_schema)
            return list(validator_cls(json_schema).iter_errors(payload))
        except Exception as exc:  # schema 自身问题不应影响正常生成
            logger.debug("Skipping schema validation for Qwen output: %s", exc)
            return []

    @staticmethod
    def _format_violation(error: Any) -> str:
        location = "$" + "".join(
            f"[{part!r}]" if isinstance(part, str) else f"[{part}]"
            for part in error.absolute_path
        )
        return f"{location}: {error.message}"

    @classmethod
    def _schema_violations(cls, payload: Any, json_schema: Dict[str, Any]) -> List[str]:
        """List schema violations as human-readable paths, or [] when unverifiable."""
        violations = [
            cls._format_violation(error)
            for error in cls._iter_schema_errors(payload, json_schema)
        ]
        return violations[:_MAX_REPORTED_SCHEMA_VIOLATIONS]

    @classmethod
    def _truncate_overlong_strings(
        cls,
        payload: Any,
        json_schema: Dict[str, Any],
    ) -> int:
        """Trim strings that only violate `maxLength`, in place. Returns the trim count.

        maxLength 是模型无法可靠自检的约束（它得边写边数字符），而违规值本身完全可用，
        上限也已知，因此本地截断是确定性的零成本修复。把这类违规交给纠错重试，等于
        为一个能就地修好的问题多花一次调用——实测占 qwen schema 重试的九成以上。
        仅处理 maxLength 单一失因，其余违规（缺字段、错键名、越界枚举）仍交给模型重试。
        """
        trimmed = 0
        for error in cls._iter_schema_errors(payload, json_schema):
            if error.validator != "maxLength":
                continue
            limit = error.validator_value
            if not isinstance(limit, int) or limit < 0:
                continue
            path = list(error.absolute_path)
            if not path:
                continue
            container = payload
            try:
                for part in path[:-1]:
                    container = container[part]
                key = path[-1]
                value = container[key]
            except (KeyError, IndexError, TypeError):
                continue
            if not isinstance(value, str) or len(value) <= limit:
                continue
            container[key] = value[:limit]
            trimmed += 1
        return trimmed

    async def _enforce_schema_with_retry(
        self,
        parsed: Any,
        json_schema: Dict[str, Any],
        base_messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        request_kwargs: Dict[str, Any],
    ) -> Any:
        """Retry once with the violation list when the parsed payload breaks the schema.

        软失败语义：重试后仍不合规时返回违规较少的一方，由下游契约校验做最终裁决，
        以免把一次可恢复的字段缺失升级成整个 session 的异常。

        超长字符串先本地截断，只有截断后仍存在的结构性违规才值得花一次纠错重试。
        """
        trimmed = self._truncate_overlong_strings(parsed, json_schema)
        if trimmed:
            logger.info(
                "Trimmed %d over-length string field(s) in Qwen output to satisfy maxLength",
                trimmed,
            )

        violations = self._schema_violations(parsed, json_schema)
        if not violations:
            return parsed

        logger.warning(
            "Qwen output violates the requested schema (%d issue(s)); retrying once: %s",
            len(violations),
            "; ".join(violations),
        )
        try:
            retry_messages = [
                *base_messages,
                {
                    "role": "assistant",
                    "content": json.dumps(parsed, ensure_ascii=False),
                },
                {
                    "role": "user",
                    "content": (
                        "Your previous JSON violated the required schema:\n"
                        + "\n".join(f"- {violation}" for violation in violations)
                        + "\n\nReturn ONE complete corrected JSON object that satisfies the "
                        "schema. Include EVERY field listed in the schema's `required` arrays, "
                        "even when a value is empty — use \"\" for strings and [] for arrays "
                        "instead of omitting the key. Do not output the schema itself. "
                        "Do not explain. Do not use markdown fences."
                    ),
                },
            ]
            retry_response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=retry_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **request_kwargs,
            )
            retry_parsed = self._parse_json_result_text(
                self._normalize_json_text(self._extract_completion_text(retry_response))
            )
        except Exception as exc:
            logger.warning("Schema-corrective retry failed for Qwen output: %s", exc)
            return parsed

        # 重试载荷同样先本地截断，否则"压缩后仍略超上限"会被误判为纠错无效。
        self._truncate_overlong_strings(retry_parsed, json_schema)
        retry_violations = self._schema_violations(retry_parsed, json_schema)
        if not retry_violations:
            logger.info("Schema-corrective retry produced a conforming Qwen output")
            return retry_parsed
        if len(retry_violations) < len(violations):
            logger.warning(
                "Schema-corrective retry still violates the schema but improved "
                "(%d -> %d issue(s)); using the retry payload",
                len(violations),
                len(retry_violations),
            )
            return retry_parsed
        logger.warning(
            "Schema-corrective retry did not improve schema conformance "
            "(%d issue(s)); keeping the original payload",
            len(retry_violations),
        )
        return parsed

    async def generate_with_json_output(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        按给定 JSON schema 生成结构化输出，并尽量修复不合法 JSON。
        """
        if self.client is None:
            raise RuntimeError("Qwen client is not initialized.")

        enhanced_system_prompt = self._build_json_system_prompt(
            json_schema=json_schema,
            system_prompt=system_prompt,
        )

        try:
            request_kwargs = self._apply_thinking_disabled(kwargs)
            effective_temperature = temperature if temperature is not None else self.temperature
            effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
            base_messages = [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=base_messages,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
                **request_kwargs,
            )

            try:
                result_text = self._normalize_json_text(self._extract_completion_text(response))
                parsed = self._parse_json_result_text(result_text)
                if not self._looks_like_truncated_json(result_text):
                    return await self._enforce_schema_with_retry(
                        parsed=parsed,
                        json_schema=json_schema,
                        base_messages=base_messages,
                        temperature=effective_temperature,
                        max_tokens=effective_max_tokens,
                        request_kwargs=request_kwargs,
                    )
            except ValueError:
                looks_truncated = self._looks_like_truncated_json(result_text)
                retry_max_tokens = (
                    max(effective_max_tokens, min(effective_max_tokens * 2, 8192))
                    if looks_truncated else
                    effective_max_tokens
                )
                retry_messages = [
                    *base_messages,
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was malformed JSON. "
                            "Return one VALID JSON object instance from scratch. "
                            "Do not use markdown fences. Do not omit closing braces, brackets, or quotes."
                            " Keep string fields concise and avoid unnecessary repetition."
                        ),
                    },
                ]
                logger.warning(
                    "Detected malformed JSON from Qwen; retrying once with max_tokens=%s truncated=%s",
                    retry_max_tokens,
                    looks_truncated,
                )
                retry_response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=retry_messages,
                    temperature=effective_temperature,
                    max_tokens=retry_max_tokens,
                    **request_kwargs,
                )
                retry_result_text = self._normalize_json_text(
                    self._extract_completion_text(retry_response)
                )
                return await self._enforce_schema_with_retry(
                    parsed=self._parse_json_result_text(retry_result_text),
                    json_schema=json_schema,
                    base_messages=base_messages,
                    temperature=effective_temperature,
                    max_tokens=retry_max_tokens,
                    request_kwargs=request_kwargs,
                )
            retry_max_tokens = max(effective_max_tokens, min(effective_max_tokens * 2, 8192))
            retry_messages = [
                *base_messages,
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was truncated because it exceeded the output budget. "
                        "Return one COMPLETE, VALID JSON object from scratch with every required field, "
                        "but compress the content to fit the budget: keep every string field to at most "
                        "two short sentences, paraphrase instead of quoting long passages, and keep arrays "
                        "minimal. Do not use markdown fences. Do not omit closing braces, brackets, or quotes."
                    ),
                },
            ]
            logger.warning(
                "Detected truncated JSON from Qwen after repair; retrying once with max_tokens=%s",
                retry_max_tokens,
            )
            retry_response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=retry_messages,
                temperature=effective_temperature,
                max_tokens=retry_max_tokens,
                **request_kwargs,
            )
            retry_result_text = self._normalize_json_text(
                self._extract_completion_text(retry_response)
            )
            return await self._enforce_schema_with_retry(
                parsed=self._parse_json_result_text(retry_result_text),
                json_schema=json_schema,
                base_messages=base_messages,
                temperature=effective_temperature,
                max_tokens=retry_max_tokens,
                request_kwargs=request_kwargs,
            )
        except Exception as e:
            logger.error(f"Error generating JSON response from Qwen: {e}")
            raise

    async def generate_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        default: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        BaseModel 接口：返回 JSON，支持 default 兜底。
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        生成文本 embedding。
        """
        if self.client is None:
            raise RuntimeError("Qwen client is not initialized.")

        try:
            text_list = [text] if isinstance(text, str) else text
            embeddings = []
            for start in range(0, len(text_list), self.embedding_batch_size):
                batch = text_list[start : start + self.embedding_batch_size]
                response = await self.client.embeddings.create(
                    model=self.embedding_model_name,
                    input=batch,
                )
                embeddings.extend(item.embedding for item in response.data)
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings from Qwen: {e}")
            raise

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QwenModel":
        """
        从配置字典创建 QwenModel 实例。

        同时兼容：
        - 顶层模型配置（如 config.models.qwen_model）
        - agent 内部的 model: {...} 字段
        """
        # 如果有嵌套的 model 字段，优先使用其中的 api_base / api_key / model
        nested_model_cfg = config.get("model", {}) or {}

        api_key = _resolve_env_like_value(nested_model_cfg.get("api_key")) or _resolve_env_like_value(config.get("api_key"))
        base_url = (
            _resolve_env_like_value(nested_model_cfg.get("api_base"))
            or _resolve_env_like_value(nested_model_cfg.get("base_url"))
            or _resolve_env_like_value(config.get("api_base"))
            or _resolve_env_like_value(config.get("base_url"))
        )

        model_name = (
            nested_model_cfg.get("model")
            or config.get("model_name")
            or "qwen3.5-plus"
        )
        embedding_model_name = (
            nested_model_cfg.get("embedding_model_name")
            or config.get("embedding_model_name")
            or "text-embedding-v4"
        )
        embedding_batch_size = (
            nested_model_cfg.get("embedding_batch_size")
            or config.get("embedding_batch_size")
            or 20
        )

        max_tokens = nested_model_cfg.get("max_tokens") or config.get("max_tokens", 1024)
        temperature = nested_model_cfg.get("temperature") or config.get("temperature", 0.3)
        timeout = config.get("timeout", 600)

        return cls(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            embedding_model_name=embedding_model_name,
            embedding_batch_size=embedding_batch_size,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
