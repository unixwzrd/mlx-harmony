from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterator, List, Optional, Union

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
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

from .config import PromptConfig, apply_placeholders


class TokenGenerator:
    """
    Multi-model token generator using MLX-LM + Harmony.

    - Works with any MLX-LM supported model.
    - Automatically uses Harmony format for GPT-OSS models.
    """

    def __init__(
        self,
        model_path: str,
        use_harmony: Optional[bool] = None,
        lazy: bool = False,
        prompt_config: Optional[PromptConfig] = None,
    ) -> None:
        """
        Initialize generator for any MLX-LM model.

        Args:
            model_path: Path to model checkpoint or Hugging Face repo.
            use_harmony: Whether to use Harmony format. When None, this is
                auto-detected (enabled only for GPT-OSS models).
            lazy: Lazy-load model weights.
        """
        self.model, self.tokenizer = load(model_path, lazy=lazy)
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
            self.streamable_parser = StreamableParser.new(
                self.encoding, Role.ASSISTANT
            )

    @staticmethod
    def _is_gpt_oss_model(model_path: str) -> bool:
        """Best-effort detection of GPT-OSS models from the model identifier."""
        path_lower = model_path.lower()
        return "gpt-oss" in path_lower or "gpt_oss" in path_lower

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
            xtc_special_tokens=[],
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

        kwargs: Dict[str, object] = {"sampler": sampler}
        if logits_processors:
            kwargs["logits_processors"] = logits_processors
        if max_tokens is not None and max_tokens > 0:
            kwargs["max_tokens"] = max_tokens

        if stop_tokens:
            stop_strings = self._tokens_to_stop_strings(stop_tokens)
            if stop_strings:
                kwargs["stop"] = stop_strings

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt_str,
            **kwargs,
        ):
            token_id = self._extract_token_id(response)

            if return_logprobs:
                yield {
                    "token": token_id,
                    # MLX-LM Response currently does not expose per-token logprobs
                    # on the public API; keep this for future extension.
                    "logprob": getattr(response, "logprob", None),
                }
            else:
                yield token_id

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

        for msg in messages:
            role_str = msg.get("role", "user").upper()

            # Handle tool messages: Role.TOOL with name in Author.name
            # Only treat as tool message if role is explicitly "tool"
            if role_str == "TOOL":
                tool_name = msg.get("name")
                if not tool_name:
                    # Tool messages require a name field; skip invalid message
                    continue
                author = Author(role=Role.TOOL, name=tool_name)
                content_text = msg.get("content", "")
                content = TextContent.new().with_text(content_text)
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
                content = TextContent.new().with_text(content_text)
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
