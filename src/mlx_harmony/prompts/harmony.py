from __future__ import annotations

from datetime import datetime
from typing import Optional

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    TextContent,
)

from mlx_harmony.config import PromptConfig, apply_placeholders


class HarmonyPromptRenderer:
    """Render prompts using Harmony encoding for GPT-OSS style models."""

    def __init__(self, *, encoding, prompt_config: PromptConfig | None) -> None:
        self.encoding = encoding
        self.prompt_config = prompt_config
        self._default_conversation_start_date = datetime.now().strftime("%Y-%m-%d")

    def render_prompt_text(
        self, messages: list[dict[str, str]], system_message: Optional[str]
    ) -> str:
        conversation = self._build_conversation(messages, system_message)
        prompt_tokens = self.encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )
        return self.encoding.decode(prompt_tokens)

    def render_prompt_tokens(
        self, messages: list[dict[str, str]], system_message: Optional[str]
    ) -> list[int]:
        conversation = self._build_conversation(messages, system_message)
        prompt_tokens = self.encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )
        assistant_header = self.encoding.encode(
            "<|start|>assistant", allowed_special={"<|start|>"}
        )
        if (
            len(prompt_tokens) >= len(assistant_header)
            and prompt_tokens[-len(assistant_header):] == assistant_header
        ):
            channel_header = self.encoding.encode(
                "<|channel|>analysis<|message|>",
                allowed_special={"<|channel|>", "<|message|>"},
            )
            prompt_tokens = prompt_tokens + channel_header
        return prompt_tokens

    def _build_conversation(
        self,
        messages: list[dict[str, str]],
        system_message: Optional[str],
    ) -> Conversation:
        system_override: Optional[str] = None
        developer_override: Optional[str] = None
        for msg in messages:
            role_str = msg.get("role", "user").strip().lower()
            if role_str == "system":
                system_override = msg.get("content", "")
            elif role_str == "developer":
                developer_override = msg.get("content", "")

        if system_override or developer_override:
            messages = [
                msg
                for msg in messages
                if msg.get("role", "user").strip().lower()
                not in ("system", "developer")
            ]

        harmony_messages: list[Message] = []
        sys_content = SystemContent.new()
        cfg = self.prompt_config

        if system_message:
            sys_content = sys_content.with_model_identity(system_message)
        elif system_override:
            sys_content = sys_content.with_model_identity(system_override)
        elif cfg and cfg.system_model_identity:
            sys_content = sys_content.with_model_identity(
                apply_placeholders(cfg.system_model_identity, cfg.placeholders)
                if cfg.placeholders
                else cfg.system_model_identity
            )

        if cfg and cfg.reasoning_effort:
            try:
                effort = ReasoningEffort(cfg.reasoning_effort.capitalize())
                sys_content = sys_content.with_reasoning_effort(effort)
            except ValueError:
                pass

        if cfg and cfg.conversation_start_date:
            sys_content = sys_content.with_conversation_start_date(
                cfg.conversation_start_date
            )
        else:
            sys_content = sys_content.with_conversation_start_date(
                self._default_conversation_start_date
            )

        if cfg and cfg.knowledge_cutoff:
            sys_content = sys_content.with_knowledge_cutoff(cfg.knowledge_cutoff)

        harmony_messages.append(
            Message.from_role_and_content(Role.SYSTEM, sys_content),
        )

        if developer_override:
            dev_content = DeveloperContent.new().with_instructions(developer_override)
            harmony_messages.append(Message.from_role_and_content(Role.DEVELOPER, dev_content))
        elif cfg and cfg.developer_instructions:
            instructions = (
                apply_placeholders(cfg.developer_instructions, cfg.placeholders)
                if cfg.placeholders
                else cfg.developer_instructions
            )
            dev_content = DeveloperContent.new().with_instructions(instructions)
            harmony_messages.append(
                Message.from_role_and_content(Role.DEVELOPER, dev_content),
            )

        if cfg and cfg.example_dialogues:
            for example_turns in cfg.example_dialogues:
                for turn in example_turns:
                    role_str = turn.get("role", "user").strip().lower()
                    if role_str == "tool":
                        tool_name = turn.get("name")
                        if tool_name:
                            author = Author(role=Role.TOOL, name=tool_name)
                            content = TextContent(text=turn.get("content", ""))
                            tool_msg = Message.from_author_and_content(author, content)
                            if turn.get("recipient"):
                                tool_msg = tool_msg.with_recipient(turn["recipient"])
                            harmony_messages.append(tool_msg)
                    else:
                        role = Role(role_str)
                        content = TextContent(text=turn.get("content", ""))
                        harmony_messages.append(
                            Message.from_role_and_content(role, content),
                        )

        for msg in messages:
            role_str = msg.get("role", "user").strip().lower()
            if role_str == "tool":
                tool_name = msg.get("name")
                if not tool_name:
                    continue
                author = Author(role=Role.TOOL, name=tool_name)
                content_text = msg.get("content", "")
                content = TextContent(text=content_text)
                tool_msg = Message.from_author_and_content(author, content)
                if msg.get("recipient"):
                    tool_msg = tool_msg.with_recipient(msg["recipient"])
                if msg.get("channel"):
                    tool_msg = tool_msg.with_channel(msg["channel"])
                harmony_messages.append(tool_msg)
            else:
                role = Role(role_str)
                content_text = msg.get("content", "")
                content = TextContent(text=content_text)
                harmony_messages.append(
                    Message.from_role_and_content(role, content),
                )

        return Conversation.from_messages(harmony_messages)
