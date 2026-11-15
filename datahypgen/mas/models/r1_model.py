"""
DeepSeek R1 Model Adapter for DataGenHyp

This module implements the BaseModel interface for DeepSeek R1 reasoning models.
DeepSeek R1 models provide advanced reasoning capabilities with explicit thinking
processes that are separated from the final answer using XML-style tags.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Any, Union

from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)

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


class R1Model(BaseModel):
    """
    DeepSeek R1 model implementation with reasoning capabilities.

    This model adapter interfaces with DeepSeek R1 models that provide explicit
    reasoning traces. The model outputs its thinking process wrapped in <think> tags
    followed by the final answer. This implementation automatically extracts and
    returns only the answer portion while discarding the thinking trace.

    The model is compatible with the OpenAI API format and uses AsyncOpenAI client
    for async operations.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                base_url: Optional[str] = None,
                model_name: str = "DeepSeek-R1",
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 300,
                extra_body: Optional[Dict[str, Any]] = None,
                response_format: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenAI model adapter.
        
        Args:
            api_key: OpenAI API key (defaults to DS_API_KEY environment variable)
            model_name: Model name to use (e.g., "DeepSeek-R1")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
            extra_body: Extra request body passed to chat completions
            response_format: Response format passed to chat completions
        """
        self.api_key = _resolve_env_like_value(api_key) or os.environ.get("DS_API_KEY")
        if not self.api_key:
            logger.warning("DS API key not provided. Please set DS_API_KEY environment variable.")
        self.base_url = _resolve_env_like_value(base_url) or os.environ.get("DS_API_BASE_URL")
        if not self.base_url:
            logger.warning("DS base URL not provided. Please set DS_API_BASE_URL environment variable.")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.extra_body = dict(extra_body) if isinstance(extra_body, dict) else None
        self.response_format = dict(response_format) if isinstance(response_format, dict) else None
        
        # Initialize the client with only the supported parameters for version 1.3.3
        try:
            client_kwargs = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": 0,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self.client = AsyncOpenAI(**client_kwargs)
            logger.info(
                "DeepSeek R1 client initialized with model: %s via %s",
                self.model_name,
                self.base_url or "provider_default",
            )
        except TypeError as e:
            logger.warning(f"Error initializing DeepSeek client: {e}")
            self.client = None

    def _build_request_kwargs(
        self,
        kwargs: Dict[str, Any],
        *,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        request_kwargs = dict(kwargs)

        configured_extra_body = self.extra_body
        runtime_extra_body = request_kwargs.get("extra_body")
        if isinstance(configured_extra_body, dict) or isinstance(runtime_extra_body, dict):
            merged_extra_body = {}
            if isinstance(configured_extra_body, dict):
                merged_extra_body.update(configured_extra_body)
            if isinstance(runtime_extra_body, dict):
                merged_extra_body.update(runtime_extra_body)
            request_kwargs["extra_body"] = merged_extra_body

        effective_response_format = response_format or self.response_format
        if effective_response_format is not None and "response_format" not in request_kwargs:
            request_kwargs["response_format"] = effective_response_format

        return request_kwargs

    def _extract_message_content(self, response: Any) -> str:
        choice = response.choices[0]
        content = choice.message.content
        if content is None:
            raise ValueError("DeepSeek response did not include message.content")
        if "</think>" in content:
            _, answer_text = content.split("</think>", 1)
            return answer_text.lstrip("\n")
        return content

    def _extract_finish_reason(self, response: Any) -> Optional[str]:
        try:
            return response.choices[0].finish_reason
        except Exception:
            return None

    def _looks_like_truncated_json(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return True
        if not stripped.endswith(("}", "]")):
            return True
        return stripped.count("{") > stripped.count("}") or stripped.count("[") > stripped.count("]")
    
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
        request_kwargs = self._build_request_kwargs(kwargs)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **request_kwargs
            )
            
            return self._extract_message_content(response)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating response from OpenAI: {error_msg}")
            logger.error(f"  Model: {self.model_name}")
            logger.error(f"  Base URL: {self.base_url}")
            logger.error(f"  Timeout: {self.timeout}")
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
                                       max_tokens: Optional[int] = None,
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

        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\nRespond with JSON that matches this schema: {json.dumps(json_schema)}"
        else:
            enhanced_system_prompt = f"Respond with JSON that matches this schema: {json.dumps(json_schema)}"

        try:
            request_kwargs = self._build_request_kwargs(
                kwargs,
                response_format={"type": "json_object"},
            )
            effective_temperature = temperature if temperature is not None else self.temperature
            effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
            messages = [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
                **request_kwargs
            )
            
            answer_text = self._extract_message_content(response)
            try:
                return json.loads(answer_text)
            except json.JSONDecodeError:
                finish_reason = self._extract_finish_reason(response)
                looks_truncated = self._looks_like_truncated_json(answer_text)
                logger.warning(
                    "DeepSeek JSON decode failed: finish_reason=%s response_chars=%s truncated=%s",
                    finish_reason,
                    len(answer_text),
                    looks_truncated,
                )
                if finish_reason == "length" or looks_truncated:
                    retry_max_tokens = max(effective_max_tokens, min(effective_max_tokens * 2, 16384))
                    logger.warning(
                        "Retrying DeepSeek JSON generation once with max_tokens=%s",
                        retry_max_tokens,
                    )
                    retry_response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            *messages,
                            {
                                "role": "user",
                                "content": (
                                    "Your previous reply was truncated or malformed JSON. "
                                    "Return one COMPLETE valid JSON object from scratch. "
                                    "Do not use markdown fences. Keep string fields concise."
                                ),
                            },
                        ],
                        temperature=effective_temperature,
                        max_tokens=retry_max_tokens,
                        **request_kwargs
                    )
                    retry_answer_text = self._extract_message_content(retry_response)
                    return json.loads(retry_answer_text)
                raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating JSON response from OpenAI: {error_msg}")
            logger.error(f"  Model: {self.model_name}")
            logger.error(f"  Base URL: {self.base_url}")
            logger.error(f"  Timeout: {self.timeout}")
            logger.error(f"  API Key present: {bool(self.api_key)}")
            if hasattr(e, 'status_code'):
                logger.error(f"  HTTP Status: {e.status_code}")
            if hasattr(e, 'response'):
                logger.error(f"  Response: {e.response}")
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
                model="text-embedding-ada-002",
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'R1Model':
        """
        Create an OpenAI model instance from a configuration dictionary.

        This factory method enables consistent model instantiation across the system
        based on configuration settings.
        
        Args:
            config: Debug configuration   dictionary with model settings
            
        Returns:
            Configured OpenAIModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            base_url=(
                _resolve_env_like_value(config.get("api_base"))
                or _resolve_env_like_value(config.get("base_url"))
            ),
            model_name=config.get("model_name", "DeepSeek-R1"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60),
            extra_body=config.get("extra_body"),
            response_format=config.get("response_format")
        ) 
