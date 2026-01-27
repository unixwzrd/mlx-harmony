from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Union

from openai_harmony import (
    HarmonyEncodingName,
    Role,
    StreamableParser,
    load_harmony_encoding,
)

from mlx_harmony.config import PromptConfig, apply_placeholders
from mlx_harmony.generation.backends import GPTOSSBackend, NativeBackend
from mlx_harmony.hyperparameters import resolve_param
from mlx_harmony.prompts.harmony import HarmonyPromptRenderer
from mlx_harmony.prompts.native import NativePromptRenderer


class TokenGenerator:
    """
    Multi-model token generator using MLX + Harmony.

    - Works with any MLX-compatible model (uses mlx-lm model architectures only).
    - Standalone generation and sampling implementation.
    - Automatically uses Harmony format for GPT-OSS models.
    """

    def __init__(
        self,
        model_path: str,
        use_harmony: Optional[bool] = None,
        lazy: bool = False,
        prompt_config: Optional[PromptConfig] = None,
        mlock: bool = False,
    ) -> None:
        """
        Initialize generator for any MLX-compatible model.

        Args:
            model_path: Path to model checkpoint or Hugging Face repo.
            use_harmony: Whether to use Harmony format. When None, this is
                auto-detected (enabled only for GPT-OSS models).
            lazy: Lazy-load model weights.
            mlock: If True, lock model weights in memory using MLX's wired limit
                (mlock equivalent, macOS Metal only). Default: False
        """
        # Use standalone loader (no pre-warming, filesystem cache handles it naturally)
        # Imported lazily to avoid initializing MLX during module import.
        from mlx_harmony.loader import load_model_standalone

        self.model, self.tokenizer = load_model_standalone(
            model_path,
            lazy=lazy,
            mlock=mlock,
        )
        self.model_path = model_path
        self.prompt_config = prompt_config
        self.mlock = mlock
        self._prompt_cache: list[object] | None = None
        self._prompt_cache_tokens: list[int] | None = None
        self._prompt_cache_max_kv_size: int | None = None
        self._last_prefill_start_offset: int | None = None

        # Auto-detect if this is a GPT-OSS model.
        self.is_gpt_oss = self._is_gpt_oss_model(model_path)

        if use_harmony is None:
            use_harmony = self.is_gpt_oss
        # Only enable Harmony when both requested and the model is GPT-OSS.
        self.use_harmony = bool(use_harmony and self.is_gpt_oss)

        self.encoding = (
            load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            if self.use_harmony
            else None
        )
        self.prompt_renderer = (
            HarmonyPromptRenderer(encoding=self.encoding, prompt_config=prompt_config)
            if self.use_harmony and self.encoding is not None
            else NativePromptRenderer(tokenizer=self.tokenizer)
        )
        self.backend = (
            GPTOSSBackend(
                encoding=self.encoding,
                tokenizer=self.tokenizer,
                prompt_config=prompt_config,
            )
            if self.use_harmony and self.encoding is not None
            else NativeBackend(tokenizer=self.tokenizer)
        )
        self.last_finish_reason: str | None = None
        self.last_stop_token_id: int | None = None
        self.last_stop_reason: str | None = None

        # Streamable parser for tool call detection (Harmony only)
        self.streamable_parser: Optional[StreamableParser] = None
        if self.use_harmony and self.encoding is not None:
            # StreamableParser constructor takes encoding and optional role
            # Use strict=False for permissive parsing (allows recovery from malformed output)
            self.streamable_parser = StreamableParser(self.encoding, Role.ASSISTANT, strict=False)

    @staticmethod
    def _is_gpt_oss_model(model_path: str) -> bool:
        """Best-effort detection of GPT-OSS models from the model identifier."""
        path_lower = model_path.lower()
        return "gpt-oss" in path_lower or "gpt_oss" in path_lower

    def _resolve_xtc_special_tokens(
        self, config_tokens: Optional[List[int]], xtc_probability: float
    ) -> list[int]:
        """
        Resolve XTC special tokens from config or auto-detect from tokenizer.

        If config_tokens is None and XTC is enabled (xtc_probability > 0.0),
        auto-detects special tokens (EOS and newline) from the tokenizer.
        Otherwise, returns the configured tokens or empty list.
        """
        # If explicitly set in config, use that
        if config_tokens is not None:
            return config_tokens

        # If XTC is disabled, return empty list
        if xtc_probability <= 0.0:
            return []

        # XTC is enabled but no special tokens specified - auto-detect from tokenizer
        # Similar to mlx-lm's server.py, we exclude EOS and newline tokens
        special_tokens: list[int] = []
        try:
            # Add EOS token if available
            if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                eos_id = self.tokenizer.eos_token_id
                if isinstance(eos_id, int):
                    special_tokens.append(eos_id)
                elif isinstance(eos_id, list) and eos_id:
                    # Some tokenizers have eos_token_id as a list
                    special_tokens.extend(eos_id)

            # Add newline token if available
            newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            if newline_ids:
                special_tokens.extend(list(newline_ids))

            # Remove duplicates while preserving order
            seen = set()
            unique_tokens = []
            for token_id in special_tokens:
                if token_id not in seen:
                    seen.add(token_id)
                    unique_tokens.append(token_id)

            return unique_tokens
        except Exception:
            # If auto-detection fails, return empty list (no special tokens excluded)
            return []

    def generate(
        self,
        prompt_tokens: Optional[List[int]] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        stop_tokens: Optional[List[int]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_tokens_to_keep: Optional[int] = None,
        xtc_probability: Optional[float] = None,
        xtc_threshold: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        return_logprobs: bool = False,
        system_message: Optional[str] = None,
        seed: Optional[int] = None,
        clear_cache: Optional[bool] = None,
        clear_cache_interval: Optional[int] = None,
        log_memory_stats: Optional[bool] = None,
        log_timing_stats: Optional[bool] = None,
        loop_detection: Optional[str] = None,
        clear_cache_generation: Optional[bool] = None,
    ) -> Iterator[Union[int, Dict[str, Union[int, float, None]]]]:
        """
        Generate tokens with automatic format selection.

        - GPT-OSS models: Uses Harmony format.
        - Other models: Uses the model's native chat template.
        """
        # Imported lazily to avoid MLX initialization during module import.
        from mlx_harmony.generate_standalone import stream_generate
        from mlx_harmony.sampling import build_logits_processors, make_sampler

        # Use provided prompt token IDs when available (avoids re-rendering prompts).
        prompt_input, prompt_token_list = self.backend.prepare_prompt(
            prompt_tokens=prompt_tokens,
            messages=messages,
            prompt=prompt,
            system_message=system_message,
        )
        self.last_finish_reason = None
        self.last_stop_token_id = None
        self.last_stop_reason = None

        cfg = self.prompt_config

        if seed is not None and seed >= 0:
            # Lazy import to avoid MLX initialization during module import.
            import mlx.core as mx

            mx.random.seed(seed)

        resolved_temp = resolve_param(temperature, cfg.temperature if cfg else None, 1.0)
        sampler_is_greedy = resolved_temp <= 0.0
        # Build sampler using MLX-LM's sampling utilities.
        sampler = make_sampler(
            temp=resolved_temp,
            top_p=resolve_param(top_p, cfg.top_p if cfg else None, 0.0),
            min_p=resolve_param(min_p, cfg.min_p if cfg else None, 0.0),
            min_tokens_to_keep=resolve_param(
                min_tokens_to_keep, cfg.min_tokens_to_keep if cfg else None, 1
            ),
            top_k=resolve_param(top_k, cfg.top_k if cfg else None, 0),
            xtc_probability=resolve_param(
                xtc_probability, cfg.xtc_probability if cfg else None, 0.0
            ),
            xtc_threshold=resolve_param(
                xtc_threshold, cfg.xtc_threshold if cfg else None, 0.0
            ),
            xtc_special_tokens=self._resolve_xtc_special_tokens(
                cfg.xtc_special_tokens if cfg else None,
                resolve_param(xtc_probability, cfg.xtc_probability if cfg else None, 0.0),
            ),
        )

        default_repetition_context = 256 if self.use_harmony else 20
        default_repetition_penalty = 1.05 if self.use_harmony else 0.0
        resolved_repetition_penalty = resolve_param(
            repetition_penalty,
            cfg.repetition_penalty if cfg else None,
            default_repetition_penalty,
        )
        repetition_penalty_value = (
            resolved_repetition_penalty
            if resolved_repetition_penalty is not None
            and resolved_repetition_penalty != 1.0
            and resolved_repetition_penalty != 0.0
            else None
        )
        logits_processors = build_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=repetition_penalty_value,
            repetition_context_size=resolve_param(
                repetition_context_size,
                cfg.repetition_context_size if cfg else None,
                default_repetition_context,
            ),
        )

        # Resolve max_tokens: CLI/function arg > prompt_config > default
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None and cfg:
            perf_mode = bool(getattr(cfg, "performance_mode", False))
            perf_max_tokens = getattr(cfg, "perf_max_tokens", None)
            if perf_mode and perf_max_tokens is not None:
                resolved_max_tokens = perf_max_tokens
            elif cfg.max_tokens is not None:
                resolved_max_tokens = cfg.max_tokens
        # Default max_tokens: 1024 for Harmony models (to allow for both analysis and final channels),
        # 512 for other models
        if resolved_max_tokens is None:
            resolved_max_tokens = 1024 if (self.is_gpt_oss and self.use_harmony) else 512

        kwargs: Dict[str, object] = {"sampler": sampler}
        if logits_processors:
            kwargs["logits_processors"] = logits_processors
        if resolved_max_tokens > 0:
            kwargs["max_tokens"] = resolved_max_tokens

        # For Harmony models, use HarmonyEncoding.stop_tokens_for_assistant_actions()
        # to stop only on assistant boundaries (<|return|>, <|call|>).
        stop_token_ids: List[int] = []
        if self.use_harmony and self.encoding:
            harmony_stop_ids = self.encoding.stop_tokens_for_assistant_actions()
            stop_token_ids.extend(harmony_stop_ids)

        # Add user-provided stop tokens
        if stop_tokens:
            stop_token_ids.extend(stop_tokens)

        # Remove duplicates while preserving order
        seen = set()
        stop_token_ids = [x for x in stop_token_ids if x not in seen and not seen.add(x)]

        # Track generated tokens and timing
        generated_tokens: List[int] = []
        start_time = time.perf_counter()

        prefill_start_offset = 0
        prompt_cache = None
        if prompt_token_list is not None:
            from mlx_harmony.cache import make_prompt_cache

            max_kv_size: int | None = None
            if cfg:
                perf_mode = bool(getattr(cfg, "performance_mode", False))
                perf_max_kv_size = getattr(cfg, "perf_max_kv_size", None)
                if perf_mode and perf_max_kv_size is not None:
                    max_kv_size = int(perf_max_kv_size)
                elif cfg.max_kv_size is not None:
                    max_kv_size = int(cfg.max_kv_size)

            if (
                self._prompt_cache is not None
                and self._prompt_cache_tokens is not None
                and self._prompt_cache_max_kv_size == max_kv_size
            ):
                cached_tokens = self._prompt_cache_tokens
                max_common = min(len(cached_tokens), len(prompt_token_list))
                common_prefix = 0
                while (
                    common_prefix < max_common
                    and cached_tokens[common_prefix] == prompt_token_list[common_prefix]
                ):
                    common_prefix += 1
                if common_prefix > 0:
                    prompt_cache = self._prompt_cache
                    prefill_start_offset = common_prefix
            if prompt_cache is None:
                prompt_cache = make_prompt_cache(self.model, max_kv_size=max_kv_size)

        effective_clear_cache = resolve_param(
            clear_cache,
            cfg.clear_cache if cfg else None,
            True,
        )
        if cfg and cfg.mlock:
            effective_clear_cache = False
        if self.mlock:
            effective_clear_cache = False

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt_input,
            sampler=sampler,
            logits_processors=logits_processors if logits_processors else None,
            max_tokens=resolved_max_tokens,
            stop_tokens=stop_token_ids if stop_token_ids else None,
            prompt_cache=prompt_cache,
            prefill_start_offset=prefill_start_offset,
            clear_cache=effective_clear_cache,
            clear_cache_interval=resolve_param(
                clear_cache_interval, cfg.clear_cache_interval if cfg else None, 1
            ),
            clear_cache_generation=resolve_param(
                clear_cache_generation,
                cfg.clear_cache_generation if cfg else None,
                False,
            ),
            log_memory_stats=resolve_param(
                log_memory_stats, cfg.log_memory_stats if cfg else None, False
            ),
            log_timing_stats=resolve_param(
                log_timing_stats, cfg.log_timing_stats if cfg else None, False
            ),
            loop_detection=resolve_param(
                loop_detection, cfg.loop_detection if cfg else None, "cheap"
            ),
            decode_tokens=False,
            sampler_is_greedy=sampler_is_greedy,
            compute_logprobs=return_logprobs,
        ):
            token_id = response.token

            # Check if this is a stop response (generation ended)
            if response.finish_reason == "stop":
                # Stop token was generated, don't yield it
                self.last_finish_reason = response.finish_reason
                self.last_stop_reason = response.stop_reason
                if response.stop_reason is None:
                    self.last_stop_token_id = int(token_id)
                else:
                    self.last_stop_token_id = None
                break

            if response.finish_reason == "length":
                self.last_finish_reason = response.finish_reason

            # Safe to yield - append to tracking list and yield
            generated_tokens.append(int(token_id))

            # Yield token with timing info
            if return_logprobs:
                # Defensive check: ensure logprobs exists and token_id is in it
                logprob = None
                if response.logprobs is not None:
                    # Check if token_id is accessible in logprobs
                    # logprobs might be a dict keyed by token_id, or an array
                    if isinstance(response.logprobs, dict):
                        if token_id in response.logprobs:
                            logprob_val = response.logprobs[token_id]
                            # MLX arrays can be directly converted to Python scalars (no .item() needed)
                            logprob = float(logprob_val) if hasattr(logprob_val, '__float__') else logprob_val
                    elif hasattr(response.logprobs, '__getitem__'):
                        # Assume it's array-like and token_id is an index
                        try:
                            logprob_val = response.logprobs[token_id]
                            # MLX arrays can be directly converted to Python scalars (no .item() needed)
                            logprob = float(logprob_val) if hasattr(logprob_val, '__float__') else logprob_val
                        except (IndexError, KeyError, TypeError):
                            pass

                yield {
                    "token": token_id,
                    "logprob": logprob,
                }
            else:
                yield token_id

        # After generation completes, keep parameters active (if mlock enabled)
        # This ensures buffers stay wired and don't get swapped out
        self.keepalive()

        if prompt_token_list is not None and prompt_cache is not None:
            self._prompt_cache = prompt_cache
            self._prompt_cache_tokens = prompt_token_list
            self._prompt_cache_max_kv_size = max_kv_size
            self._last_prefill_start_offset = prefill_start_offset

        # Calculate and store generation stats
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        num_tokens = len(generated_tokens)
        tokens_per_second = num_tokens / elapsed_time if elapsed_time > 0 else 0.0

        # Store stats on model for access by caller
        self._last_generation_stats = {
            "start_time": start_time,
            "end_time": end_time,
            "elapsed_time": elapsed_time,
            "num_tokens": num_tokens,
            "tokens_per_second": tokens_per_second,
        }

    def _prepare_prompt(
        self,
        prompt_tokens: Optional[List[int]] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        system_message: Optional[str] = None,
    ) -> str:
        """Prepare a text prompt from tokens/messages/plain text."""
        if messages:
            return self.render_prompt(messages, system_message)
        # Check if prompt_tokens is actually a string (common mistake when passing positional arg)
        if prompt_tokens is not None:
            if isinstance(prompt_tokens, str):
                # User passed string as positional argument, treat as prompt
                prompt = prompt_tokens
                prompt_tokens = None
            elif isinstance(prompt_tokens, (list, tuple)) and all(isinstance(x, int) for x in prompt_tokens):
                # Valid prompt_tokens (list of ints)
                # Use HarmonyEncoding.decode() if Harmony is enabled, otherwise use native tokenizer
                if self.use_harmony and self.encoding:
                    return self.encoding.decode(prompt_tokens)
                return self.tokenizer.decode(prompt_tokens)
        if prompt is not None:
            if self.use_harmony:
                messages = [{"role": "user", "content": prompt}]
                return self.render_prompt(messages, system_message)
            return prompt
        raise ValueError("Must provide prompt_tokens, messages, or prompt")

    def render_prompt(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
    ) -> str:
        """
        Convert messages to a model prompt using the appropriate format.

        Public method for rendering prompts (e.g., for debug output).
        Previously _messages_to_prompt, made public to avoid brittle private calls.
        """
        return self.prompt_renderer.render_prompt_text(messages, system_message)

    def render_prompt_tokens(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
    ) -> List[int]:
        """
        Convert messages to token IDs using the appropriate format.
        Uses Harmony encoding tokens for GPT-OSS and native tokenizer tokens otherwise.
        """
        return self.prompt_renderer.render_prompt_tokens(messages, system_message)

    def _tokens_to_stop_strings(self, stop_tokens: List[int]) -> List[str]:
        """Convert stop token IDs to stop strings."""
        stop_strings: List[str] = []
        for token_id in stop_tokens:
            # Use HarmonyEncoding.decode() if Harmony is enabled (handles special tokens correctly)
            # Otherwise use native tokenizer
            if self.use_harmony and self.encoding:
                text = self.encoding.decode([token_id])
            else:
                text = self.tokenizer.decode([token_id])
            if text:
                stop_strings.append(text)
        return stop_strings

    def keepalive(self) -> None:
        """
        Keep model parameters active to prevent deallocation (for mlock mode).

        This is a no-op if mlock is disabled or parameters are not tracked.
        Call this periodically (e.g., after generation) to prevent MLX from
        freeing parameter buffers when mlock is enabled.
        """
        if not self.mlock:
            return

        if not hasattr(self.model, "_mlx_harmony_param_refs"):
            return

        try:
            # Touch all parameter arrays to keep them active
            # This is a lightweight operation that prevents deallocation
            for param in self.model._mlx_harmony_param_refs:
                # Access the array data to prevent deallocation
                _ = param.shape
        except Exception:
            # Ignore errors - not critical
            pass

    def parse_messages_from_tokens(
        self,
        tokens: List[int],
        *,
        strict: bool = True,
    ) -> List[Message]:
        """
        Parse Harmony messages from completion tokens.

        Only works when Harmony is enabled (GPT-OSS models).
        """
        if not self.use_harmony or self.encoding is None:
            raise ValueError(
                "parse_messages_from_tokens only works with Harmony encoding"
            )
        return self.encoding.parse_messages_from_completion_tokens(
            tokens, Role.ASSISTANT, strict=strict
        )
