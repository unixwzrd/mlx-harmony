from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai_harmony import Role
from unicodefix.transforms import clean_text

from mlx_harmony.logging import get_logger

logger = get_logger(__name__)


def _decode_tokens(encoding: Any, tokens: list[int]) -> str:
    decode_utf8 = getattr(encoding, "decode_utf8", None)
    if callable(decode_utf8):
        return decode_utf8(tokens)
    return encoding.decode(tokens)


@dataclass(frozen=True)
class HarmonyParseResult:
    assistant_text: str
    analysis_text_parts: list[str]
    parsed_messages: list[Any] | None


def parse_harmony_response(
    *,
    generator: Any,
    tokens: list[int],
    streamed_text_parts: list[str],
    assistant_name: str,
    thinking_limit: int,
    response_limit: int,
    render_markdown: bool,
    debug: bool,
    display_assistant: Callable[[str, str, bool], None],
    display_thinking: Callable[[str, bool], None],
    truncate_text: Callable[[str, int], str],
    suppress_display: bool = False,
) -> HarmonyParseResult:
    final_text_parts: list[str] = []
    analysis_text_parts: list[str] = []
    assistant_text = ""
    parsed_messages: list[Any] | None = None

    parse_tokens = tokens
    finish_reason = getattr(generator, "last_finish_reason", None)
    if (
        finish_reason == "stop"
        and getattr(generator, "last_stop_token_id", None) is not None
        and (not tokens or tokens[-1] != generator.last_stop_token_id)
        and generator.last_stop_token_id not in tokens
    ):
        parse_tokens = tokens + [generator.last_stop_token_id]

    header_tokens: list[int] | None = None
    parse_strict = True
    if generator.encoding and 200005 not in parse_tokens[:20]:
        header_tokens = generator.encoding.encode(
            "<|start|>assistant<|channel|>analysis<|message|>",
            allowed_special={"<|start|>", "<|channel|>", "<|message|>"},
        )
        parse_tokens = header_tokens + parse_tokens
        parse_strict = False
        if debug:
            logger.warning(
                "Harmony parsing with prepended assistant header (analysis channel) for completion-only parse"
            )

    try:
        parsed_messages = generator.parse_messages_from_tokens(parse_tokens, strict=parse_strict)
    except Exception as e:
        error_msg = str(e)
        if generator.encoding:
            try:
                parsed_messages = generator.parse_messages_from_tokens(
                    parse_tokens,
                    strict=False,
                )
            except Exception:
                parsed_messages = None
            else:
                if debug:
                    logger.warning(
                        "Harmony parsing retry succeeded with strict=False"
                    )
        if parsed_messages is not None:
            pass
        else:
            if debug:
                logger.error("Harmony parsing failed: %s", error_msg)
                logger.error("Tokens processed: %d", len(parse_tokens))
                if generator.encoding:
                    first_tokens = parse_tokens[:50]
                    decoded_start = _decode_tokens(generator.encoding, first_tokens)
                    logger.error("First 50 tokens decoded: %s", decoded_start[:500])
                    logger.error("First 20 token IDs: %s", parse_tokens[:20])
                    channel_token_id = 200005
                    if channel_token_id in parse_tokens[:20]:
                        idx = parse_tokens[:20].index(channel_token_id)
                        logger.error("Found <|channel|> token (200005) at position %d", idx)
                    else:
                        logger.error("<|channel|> token (200005) NOT found in first 20 tokens")
            else:
                logger.warning("Harmony parsing failed: %s", error_msg)
            if generator.encoding:
                raw_text = _decode_tokens(generator.encoding, tokens)
                raw_text = clean_text(raw_text).strip()
                if raw_text:
                    logger.warning(
                        "Harmony parsing fallback: treating completion as raw assistant text"
                    )
                    if debug:
                        logger.debug(
                            "Harmony raw fallback completion: %s",
                            raw_text[:4000],
                        )
                    assistant_text = truncate_text(raw_text, response_limit)
                    if assistant_text and not suppress_display:
                        display_assistant(assistant_text, assistant_name, render_markdown)
                    return HarmonyParseResult(
                        assistant_text=assistant_text,
                        analysis_text_parts=[],
                        parsed_messages=None,
                    )
            if debug:
                logger.error("This error indicates the model output is malformed/incomplete:")
                logger.error("  - The parser was waiting for a message header to complete")
                logger.error("  - Model output does not conform to Harmony format structure")
            raise RuntimeError(f"Failed to parse Harmony messages: {error_msg}") from e

    final_channel_messages: list[str] = []
    if debug:
        logger.debug("Parsed %d messages from parser", len(parsed_messages))
    for msg in parsed_messages:
        channel = getattr(msg, "channel", None)
        msg_text = ""
        for content in msg.content:
            if hasattr(content, "text"):
                msg_text += content.text

        if debug:
            author = getattr(msg, "author", None)
            logger.debug(
                "Message: channel=%s, author=%s, text_length=%d",
                channel,
                author,
                len(msg_text),
            )

        if channel == "final" or channel is None:
            final_channel_messages.append(clean_text(msg_text))
        elif channel in ("analysis", "commentary"):
            analysis_text_parts.append(clean_text(msg_text))

    if final_channel_messages:
        final_text_parts = [final_channel_messages[-1]]

    if analysis_text_parts:
        joined_analysis = "\n".join(analysis_text_parts)
        thinking_text = truncate_text(joined_analysis.lstrip(" \t").rstrip(" \t"), thinking_limit)
        if finish_reason == "length" and thinking_text and not thinking_text.endswith("[truncated]"):
            thinking_text = f"{thinking_text.rstrip()} ... [truncated]"
        if thinking_text and not suppress_display:
            display_thinking(thinking_text, render_markdown=render_markdown)

    if final_text_parts:
        joined_text = "".join(final_text_parts)
        assistant_text = truncate_text(joined_text.lstrip(" \t").rstrip(" \t"), response_limit)
        if finish_reason == "length" and assistant_text and not assistant_text.endswith("[truncated]"):
            assistant_text = f"{assistant_text.rstrip()} ... [truncated]"
        if assistant_text and not suppress_display:
            display_assistant(assistant_text, assistant_name, render_markdown)
    elif analysis_text_parts:
        assistant_text = ""
        if debug:
            logger.debug("Only analysis channel found in parsed_messages")
            logger.debug("Total parsed messages: %d", len(parsed_messages))
            for i, msg in enumerate(parsed_messages):
                channel = getattr(msg, "channel", None)
                logger.debug(
                    "Message %d: channel=%s, role=%s",
                    i,
                    channel,
                    getattr(msg, "author", None),
                )
    elif streamed_text_parts:
        streamed_content = "".join(streamed_text_parts).strip()
        assistant_text = truncate_text(streamed_content, response_limit)
        print()
    else:
        logger.error("Failed to parse Harmony messages: no parsed messages and no streamed content")
        logger.error("This indicates either:")
        logger.error("  - openai_harmony package is incorrectly installed")
        logger.error("  - Model output is malformed")
        logger.error("  - Parsing logic has a bug")
        if debug:
            raw_text = (
                _decode_tokens(generator.encoding, tokens)
                if generator.encoding
                else "[encoding not available]"
            )
            logger.debug("Raw decoded text: %s...", raw_text[:500])
        raise RuntimeError("Failed to parse Harmony messages: no parsed messages and no streamed content")

    return HarmonyParseResult(
        assistant_text=assistant_text,
        analysis_text_parts=analysis_text_parts,
        parsed_messages=parsed_messages,
    )
