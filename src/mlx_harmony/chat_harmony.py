from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from unicodefix.transforms import clean_text


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
) -> HarmonyParseResult:
    final_text_parts: list[str] = []
    analysis_text_parts: list[str] = []
    assistant_text = ""
    parsed_messages: list[Any] | None = None

    if generator.streamable_parser:
        parser_state = generator.streamable_parser.state
        if debug:
            print(f"\n[DEBUG] Parser state before process_eos(): {parser_state}")
            print(f"[DEBUG] Parser current_role: {generator.streamable_parser.current_role}")
            print(f"[DEBUG] Parser current_channel: {generator.streamable_parser.current_channel}")
            print(f"[DEBUG] Parser tokens processed: {len(generator.streamable_parser.tokens)}")
            if generator.streamable_parser.tokens:
                first_tokens = generator.streamable_parser.tokens[:20]
                decoded_start = generator.encoding.decode(first_tokens) if generator.encoding else "N/A"
                print(f"[DEBUG] First 20 tokens decoded: {decoded_start[:200]}")

        try:
            generator.streamable_parser.process_eos()
            parsed_messages = generator.streamable_parser.messages
        except Exception as e:
            error_msg = str(e)
            print(f"\n[ERROR] Harmony parsing failed: {error_msg}")
            print(f"[ERROR] Parser state: {parser_state}")
            print(f"[ERROR] Tokens processed: {len(generator.streamable_parser.tokens)}")
            if generator.streamable_parser.tokens and generator.encoding:
                first_tokens = generator.streamable_parser.tokens[:50]
                decoded_start = generator.encoding.decode(first_tokens)
                print(f"[ERROR] First 50 tokens decoded: {decoded_start[:500]}")
                print(f"[ERROR] First 20 token IDs: {generator.streamable_parser.tokens[:20]}")
                channel_token_id = 200005
                if channel_token_id in generator.streamable_parser.tokens[:20]:
                    idx = generator.streamable_parser.tokens[:20].index(channel_token_id)
                    print(f"[ERROR] Found <|channel|> token (200005) at position {idx}")
                else:
                    print("[ERROR] <|channel|> token (200005) NOT found in first 20 tokens")
            print("\nThis error indicates the model output is malformed/incomplete:")
            print("  - The parser was waiting for a message header to complete")
            print("  - Model output does not conform to Harmony format structure")
            raise RuntimeError(f"Failed to parse Harmony messages: {error_msg}") from e
    else:
        parsed_messages = generator.parse_messages_from_tokens(tokens)

    final_channel_messages: list[str] = []
    if debug:
        print(f"\n[DEBUG] Parsed {len(parsed_messages)} messages from parser")
    for msg in parsed_messages:
        channel = getattr(msg, "channel", None)
        msg_text = ""
        for content in msg.content:
            if hasattr(content, "text"):
                msg_text += content.text

        if debug:
            author = getattr(msg, "author", None)
            print(f"[DEBUG] Message: channel={channel}, author={author}, text_length={len(msg_text)}")

        if channel == "final" or channel is None:
            final_channel_messages.append(clean_text(msg_text))
        elif channel in ("analysis", "commentary"):
            analysis_text_parts.append(clean_text(msg_text))

    if final_channel_messages:
        final_text_parts = [final_channel_messages[-1]]

    if analysis_text_parts:
        joined_analysis = "\n".join(analysis_text_parts)
        thinking_text = truncate_text(joined_analysis.lstrip(" \t").rstrip(" \t"), thinking_limit)
        if thinking_text:
            display_thinking(thinking_text, render_markdown=render_markdown)

    if final_text_parts:
        joined_text = "".join(final_text_parts)
        assistant_text = truncate_text(joined_text.lstrip(" \t").rstrip(" \t"), response_limit)
        if assistant_text:
            display_assistant(assistant_text, assistant_name, render_markdown)
    elif analysis_text_parts:
        assistant_text = ""
        if debug:
            print("\n[DEBUG] Only analysis channel found in parsed_messages")
            print(f"[DEBUG] Total parsed messages: {len(parsed_messages)}")
            for i, msg in enumerate(parsed_messages):
                channel = getattr(msg, "channel", None)
                print(f"[DEBUG] Message {i}: channel={channel}, role={getattr(msg, 'author', None)}")
    elif streamed_text_parts:
        streamed_content = "".join(streamed_text_parts).strip()
        assistant_text = truncate_text(streamed_content, response_limit)
        print()
    else:
        print("\n[ERROR] Failed to parse Harmony messages: no parsed messages and no streamed content")
        print("This indicates either:")
        print("  - openai_harmony package is incorrectly installed")
        print("  - Model output is malformed")
        print("  - Parsing logic has a bug")
        if debug:
            raw_text = generator.encoding.decode(tokens) if generator.encoding else "[encoding not available]"
            print(f"[DEBUG] Raw decoded text: {raw_text[:500]}...")
        raise RuntimeError("Failed to parse Harmony messages: no parsed messages and no streamed content")

    return HarmonyParseResult(
        assistant_text=assistant_text,
        analysis_text_parts=analysis_text_parts,
        parsed_messages=parsed_messages,
    )
