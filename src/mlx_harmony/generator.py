from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Union

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    TextContent,
    load_harmony_encoding,
)

from mlx_harmony.config import PromptConfig, apply_placeholders
from mlx_harmony.generate_standalone import stream_generate
from mlx_harmony.loader import load_model_standalone
from mlx_harmony.sampling import make_logits_processors, make_sampler


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
        self.model, self.tokenizer = load_model_standalone(
            model_path,
            lazy=lazy,
            mlock=mlock,
        )
        self.model_path = model_path
        self.prompt_config = prompt_config

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

        # Streamable parser for tool call detection (Harmony only)
        self.streamable_parser: Optional[StreamableParser] = None
        if self.use_harmony and self.encoding is not None:
            # StreamableParser constructor takes encoding and optional role
            self.streamable_parser = StreamableParser(self.encoding, Role.ASSISTANT)

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
    ) -> Iterator[Union[int, Dict[str, Union[int, float, None]]]]:
        """
        Generate tokens with automatic format selection.

        - GPT-OSS models: Uses Harmony format.
        - Other models: Uses the model's native chat template.
        """
        prompt_str = self._prepare_prompt(
            prompt_tokens=prompt_tokens,
            messages=messages,
            prompt=prompt,
            system_message=system_message,
        )

        cfg = self.prompt_config

        def resolve(val, cfg_val, default):
            return default if val is None and cfg_val is None else (val if val is not None else cfg_val)

        # Build sampler using MLX-LM's sampling utilities.
        sampler = make_sampler(
            temp=resolve(temperature, cfg.temperature if cfg else None, 1.0),
            top_p=resolve(top_p, cfg.top_p if cfg else None, 0.0),
            min_p=resolve(min_p, cfg.min_p if cfg else None, 0.0),
            min_tokens_to_keep=resolve(
                min_tokens_to_keep, cfg.min_tokens_to_keep if cfg else None, 1
            ),
            top_k=resolve(top_k, cfg.top_k if cfg else None, 0),
            xtc_probability=resolve(
                xtc_probability, cfg.xtc_probability if cfg else None, 0.0
            ),
            xtc_threshold=resolve(
                xtc_threshold, cfg.xtc_threshold if cfg else None, 0.0
            ),
            xtc_special_tokens=self._resolve_xtc_special_tokens(
                cfg.xtc_special_tokens if cfg else None,
                resolve(xtc_probability, cfg.xtc_probability if cfg else None, 0.0),
            ),
        )

        logits_processors = make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=resolve(
                repetition_penalty,
                cfg.repetition_penalty if cfg else None,
                0.0,
            )
            if (repetition_penalty or 0.0) != 0.0
            or (cfg and cfg.repetition_penalty is not None)
            else None,
            repetition_context_size=resolve(
                repetition_context_size,
                cfg.repetition_context_size if cfg else None,
                20,
            ),
        )

        # Resolve max_tokens: CLI/function arg > prompt_config > default
        resolved_max_tokens = max_tokens
        if resolved_max_tokens is None and cfg and cfg.max_tokens is not None:
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

        # For Harmony models, automatically add Harmony stop tokens (<|return|>, <|call|>)
        # These are required for proper generation stopping - the model generates analysis,
        # then final channel, then stops with <|return|> or <|call|>
        # Note: MLX-LM's stream_generate doesn't accept 'stop' parameter, so we handle it manually
        stop_strings_list = []

        # Add user-provided stop tokens (converted to strings)
        if stop_tokens:
            stop_strings_list.extend(self._tokens_to_stop_strings(stop_tokens))

        # For Harmony models, add Harmony stop tokens
        if self.use_harmony:
            # These are the Harmony stop tokens: <|return|> (done) and <|call|> (tool call)
            harmony_stops = ["<|return|>", "<|call|>"]
            # Only add if not already present
            for stop in harmony_stops:
                if stop not in stop_strings_list:
                    stop_strings_list.append(stop)

        # Convert stop strings to token IDs (flatten all stop tokens into a single list)
        # Our standalone stream_generate handles stop tokens internally
        stop_token_ids: List[int] = []
        if stop_strings_list:
            for stop_str in stop_strings_list:
                try:
                    stop_ids = self.tokenizer.encode(stop_str, add_special_tokens=False)
                    if stop_ids:
                        # For now, add all token IDs - multi-token sequences not yet fully supported
                        # but single tokens work fine
                        stop_token_ids.extend(list(stop_ids))
                except Exception:
                    # Skip if encoding fails
                    pass

        # Track generated tokens and timing
        generated_tokens: List[int] = []
        start_time = time.perf_counter()

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt_str,
            sampler=sampler,
            logits_processors=logits_processors if logits_processors else None,
            max_tokens=resolved_max_tokens,
            stop_tokens=stop_token_ids if stop_token_ids else None,
        ):
            token_id = response.token

            # Check if this is a stop response (generation ended)
            if response.finish_reason == "stop":
                # Stop token was generated, don't yield it
                break

            # Safe to yield - append to tracking list and yield
            generated_tokens.append(int(token_id))

            # Yield token with timing info
            if return_logprobs:
                yield {
                    "token": token_id,
                    "logprob": response.logprobs[token_id].item() if response.logprobs is not None else None,
                }
            else:
                yield token_id

        # After generation completes, keep parameters active to prevent deallocation
        # This ensures buffers stay wired and don't get swapped out
        if hasattr(self.model, "_mlx_harmony_param_refs"):
            # Periodically access parameters to keep them "active" and prevent deallocation
            # This is a workaround to prevent MLX from freeing parameter buffers
            try:
                # Touch all parameter arrays to keep them active
                for param in self.model._mlx_harmony_param_refs:
                    # Access the array data to prevent deallocation
                    _ = param.shape  # Access a property to keep reference alive
            except Exception:
                pass  # Ignore errors - not critical

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
            return self._messages_to_prompt(messages, system_message)
        if prompt_tokens:
            return self.tokenizer.decode(prompt_tokens)
        if prompt is not None:
            if self.use_harmony:
                messages = [{"role": "user", "content": prompt}]
                return self._messages_to_prompt(messages, system_message)
            return prompt
        raise ValueError("Must provide prompt_tokens, messages, or prompt")

    def _messages_to_prompt(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
    ) -> str:
        """Convert messages to a model prompt using the appropriate format."""
        # Apply placeholders to message content if provided in config.
        if self.prompt_config and self.prompt_config.placeholders:
            messages = [
                {
                    **msg,
                    "content": apply_placeholders(
                        msg.get("content"), self.prompt_config.placeholders
                    ),
                }
                for msg in messages
            ]

        if self.use_harmony and self.encoding is not None:
            return self._harmony_messages_to_prompt(messages, system_message)
        return self._native_messages_to_prompt(messages, system_message)

    def _harmony_messages_to_prompt(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
    ) -> str:
        """Convert messages using Harmony format (GPT-OSS only)."""
        harmony_messages: List[Message] = []

        # Build SystemContent with sensible defaults plus optional overrides.
        sys_content = SystemContent.new()
        cfg = self.prompt_config

        # Model identity: CLI system_message wins, then config, then default.
        if system_message:
            sys_content = sys_content.with_model_identity(system_message)
        elif cfg and cfg.system_model_identity:
            sys_content = sys_content.with_model_identity(cfg.system_model_identity)

        # Reasoning effort.
        if cfg and cfg.reasoning_effort:
            try:
                effort = ReasoningEffort(cfg.reasoning_effort.capitalize())
                sys_content = sys_content.with_reasoning_effort(effort)
            except ValueError:
                # Ignore invalid value; keep default.
                pass

        # Conversation start date (config or default to today for GPT-OSS).
        if cfg and cfg.conversation_start_date:
            sys_content = sys_content.with_conversation_start_date(
                cfg.conversation_start_date
            )
        else:
            sys_content = sys_content.with_conversation_start_date(
                datetime.now().strftime("%Y-%m-%d")
            )

        # Knowledge cutoff.
        if cfg and cfg.knowledge_cutoff:
            sys_content = sys_content.with_knowledge_cutoff(cfg.knowledge_cutoff)

        # Add system message first.
        harmony_messages.append(
            Message.from_role_and_content(Role.SYSTEM, sys_content),
        )

        # Optional developer message.
        if cfg and cfg.developer_instructions:
            dev_content = DeveloperContent.new().with_instructions(
                cfg.developer_instructions
            )
            harmony_messages.append(
                Message.from_role_and_content(Role.DEVELOPER, dev_content),
            )

        # Add example dialogues (few-shot examples) before actual conversation
        # These are part of the prompt but not sent every time like system/developer
        if cfg and cfg.example_dialogues:
            for example_turns in cfg.example_dialogues:
                for turn in example_turns:
                    role_str = turn.get("role", "user").strip().lower()
                    if role_str == "tool":
                        # Handle tool messages in examples
                        tool_name = turn.get("name")
                        if tool_name:
                            author = Author(role=Role.TOOL, name=tool_name)
                            content = TextContent(text=turn.get("content", ""))
                            tool_msg = Message.from_author_and_content(author, content)
                            if turn.get("recipient"):
                                tool_msg = tool_msg.with_recipient(turn["recipient"])
                            harmony_messages.append(tool_msg)
                    else:
                        # Standard messages in examples
                        role = Role(role_str)
                        content = TextContent(text=turn.get("content", ""))
                        harmony_messages.append(
                            Message.from_role_and_content(role, content),
                        )

        for msg in messages:
            role_str = msg.get("role", "user").strip().lower()

            # Handle tool messages: Role.TOOL with name in Author.name
            # Only treat as tool message if role is explicitly "tool"
            if role_str == "tool":
                tool_name = msg.get("name")
                if not tool_name:
                    # Tool messages require a name field; skip invalid message
                    continue
                author = Author(role=Role.TOOL, name=tool_name)
                content_text = msg.get("content", "")
                content = TextContent(text=content_text)
                tool_msg = Message.from_author_and_content(author, content)
                # Set recipient if specified (tool results go to assistant)
                if msg.get("recipient"):
                    tool_msg = tool_msg.with_recipient(msg["recipient"])
                # Set channel if specified (e.g., "commentary")
                if msg.get("channel"):
                    tool_msg = tool_msg.with_channel(msg["channel"])
                harmony_messages.append(tool_msg)
            else:
                # Standard message: user, assistant, system, developer
                role = Role(role_str)
                content_text = msg.get("content", "")
                content = TextContent(text=content_text)
                harmony_messages.append(
                    Message.from_role_and_content(role, content),
                )

        conversation = Conversation.from_messages(harmony_messages)
        prompt_tokens = self.encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )
        return self.tokenizer.decode(prompt_tokens)

    def _native_messages_to_prompt(
        self,
        messages: List[Dict[str, str]],
        system_message: Optional[str] = None,
    ) -> str:
        """Convert messages using the tokenizer's native chat template."""
        if system_message:
            messages = [{"role": "system", "content": system_message}, *messages]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )

        # Fallback: simple role-prefixed text.
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def _tokens_to_stop_strings(self, stop_tokens: List[int]) -> List[str]:
        """Convert stop token IDs to stop strings."""
        stop_strings: List[str] = []
        for token_id in stop_tokens:
            text = self.tokenizer.decode([token_id])
            if text:
                stop_strings.append(text)
        return stop_strings

    def _extract_token_id(self, response: object) -> int:
        """
        Extract the last token id from an MLX-LM streaming response.

        Today the Response object exposes `.text`; we re-tokenize this text
        and take the last token id.
        """
        if hasattr(response, "text"):
            encoded = self.tokenizer.encode(response.text)
            if encoded:
                return int(encoded[-1])
        if isinstance(response, int):
            return response
        if isinstance(response, str):
            encoded = self.tokenizer.encode(response)
            if encoded:
                return int(encoded[-1])
        # Fallback - should rarely be hit.
        return 0

    def parse_messages_from_tokens(
        self, tokens: List[int]
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
            tokens, Role.ASSISTANT
        )
