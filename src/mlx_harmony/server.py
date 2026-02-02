from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from mlx_harmony.chat_adapters import get_adapter
from mlx_harmony.chat_history import (
    make_message_id,
    make_timestamp,
    write_debug_metrics,
    write_debug_response,
    write_debug_token_texts,
    write_debug_tokens,
)
from mlx_harmony.chat_prompt import (
    build_prompt_token_ids,
    prepare_prompt,
    truncate_conversation_for_context,
)
from mlx_harmony.chat_turn import run_chat_turn
from mlx_harmony.chat_utils import (
    build_hyperparameters,
    get_assistant_name,
    get_truncate_limits,
    resolve_max_context_tokens,
)
from mlx_harmony.config import (
    DEFAULT_CONFIG_DIR,
    apply_placeholders,
    load_profiles,
    load_prompt_config,
    resolve_config_path,
)
from mlx_harmony.generator import TokenGenerator
from mlx_harmony.logging import get_logger

app = FastAPI(title="MLX Harmony API")
logger = get_logger(__name__)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    messages: List[ChatMessage]
    temperature: float = 1.0
    max_tokens: int = 512
    top_p: float = 0.0
    min_p: float = 0.0
    top_k: int = 0
    repetition_penalty: float = 0.0
    repetition_context_size: int = 20
    profile: Optional[str] = None
    prompt_config: Optional[str] = None
    profiles_file: Optional[str] = None
    stream: bool = False
    return_analysis: bool = False


ChatMessage.model_rebuild()
ChatRequest.model_rebuild()


_generator: Optional[TokenGenerator] = None
_generator_prompt_config_path: Optional[str] = None
_generator_lock = Lock()
_loaded_model_path: Optional[str] = None
_loaded_at_unix: Optional[int] = None
_server_metrics_row_index = 0
_last_prompt_start_time: float | None = None

DEFAULT_PROFILES_FILE = "configs/profiles.example.json"
DEFAULT_SERVER_DEBUG_LOG = "logs/server-debug.log"
DEFAULT_SERVER_LOG_PROMPTS = "1"
DEFAULT_MODELS_DIR = "models"


def _get_generator(model: str, prompt_config_path: Optional[str]) -> TokenGenerator:
    global _generator
    global _generator_prompt_config_path
    global _loaded_model_path
    global _loaded_at_unix
    with _generator_lock:
        if (
            _generator is None
            or _generator.model_path != model
            or _generator_prompt_config_path != prompt_config_path
        ):
            prompt_cfg = load_prompt_config(prompt_config_path) if prompt_config_path else None
            logger.info(
                "Loading model: %s (prompt_config=%s)",
                model,
                prompt_config_path or "none",
            )
            _generator = TokenGenerator(model, prompt_config=prompt_cfg)
            _generator_prompt_config_path = prompt_config_path
            _loaded_model_path = model
            _loaded_at_unix = int(time.time())
        return _generator


def _resolve_profile(
    request: ChatRequest,
) -> tuple[str, Optional[str], dict[str, object] | None]:
    model_path = request.model
    prompt_config_path = request.prompt_config
    profile_data: dict[str, object] | None = None
    if request.profile:
        config_dir = os.getenv("MLX_HARMONY_CONFIG_DIR", DEFAULT_CONFIG_DIR)
        profiles_path = request.profiles_file or os.getenv(
            "MLX_HARMONY_PROFILES_FILE", DEFAULT_PROFILES_FILE
        )
        resolved_profiles_path = resolve_config_path(profiles_path, config_dir)
        if resolved_profiles_path is None or not resolved_profiles_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Profiles file not found: {profiles_path}",
            )
        profiles = load_profiles(str(resolved_profiles_path))
        if request.profile not in profiles:
            raise HTTPException(
                status_code=400,
                detail=f"Profile '{request.profile}' not found in {resolved_profiles_path}",
            )
        profile = profiles[request.profile]
        profile_data = profile
        model_path = profile.get("model", model_path)
        prompt_config_path = prompt_config_path or profile.get("prompt_config")
        if prompt_config_path:
            resolved_prompt_path = resolve_config_path(prompt_config_path, config_dir)
            if resolved_prompt_path is None or not resolved_prompt_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Prompt config not found: {prompt_config_path}",
                )
            prompt_config_path = str(resolved_prompt_path)
    elif prompt_config_path:
        config_dir = os.getenv("MLX_HARMONY_CONFIG_DIR", DEFAULT_CONFIG_DIR)
        resolved_prompt_path = resolve_config_path(prompt_config_path, config_dir)
        if resolved_prompt_path is None or not resolved_prompt_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Prompt config not found: {prompt_config_path}",
            )
        prompt_config_path = str(resolved_prompt_path)
    return model_path, prompt_config_path, profile_data


def _resolve_profile_from_args(
    *,
    model: Optional[str],
    profile: Optional[str],
    prompt_config: Optional[str],
    profiles_file: str,
) -> tuple[Optional[str], Optional[str]]:
    model_path = model
    prompt_config_path = prompt_config
    if profile:
        config_dir = os.getenv("MLX_HARMONY_CONFIG_DIR", DEFAULT_CONFIG_DIR)
        resolved_profiles_path = resolve_config_path(profiles_file, config_dir)
        if resolved_profiles_path is None or not resolved_profiles_path.exists():
            raise RuntimeError(f"Profiles file not found: {profiles_file}")
        profiles = load_profiles(str(resolved_profiles_path))
        if profile not in profiles:
            raise RuntimeError(f"Profile '{profile}' not found in {resolved_profiles_path}")
        profile_data = profiles[profile]
        model_path = profile_data.get("model", model_path)
        prompt_config_path = prompt_config_path or profile_data.get("prompt_config")
        if prompt_config_path:
            resolved_prompt_path = resolve_config_path(prompt_config_path, config_dir)
            if resolved_prompt_path is None or not resolved_prompt_path.exists():
                raise RuntimeError(f"Prompt config not found: {prompt_config_path}")
            prompt_config_path = str(resolved_prompt_path)
    elif prompt_config_path:
        config_dir = os.getenv("MLX_HARMONY_CONFIG_DIR", DEFAULT_CONFIG_DIR)
        resolved_prompt_path = resolve_config_path(prompt_config_path, config_dir)
        if resolved_prompt_path is None or not resolved_prompt_path.exists():
            raise RuntimeError(f"Prompt config not found: {prompt_config_path}")
        prompt_config_path = str(resolved_prompt_path)
    return model_path, prompt_config_path


def _decode_tokens(generator: TokenGenerator, token_ids: list[int]) -> str:
    if hasattr(generator, "tokenizer") and hasattr(generator.tokenizer, "decode"):
        return generator.tokenizer.decode(token_ids)
    backend = getattr(generator, "backend", None)
    if backend is not None and hasattr(backend, "decode"):
        return backend.decode(token_ids)
    return ""


def _next_server_metrics_row_index() -> int:
    global _server_metrics_row_index
    _server_metrics_row_index += 1
    return _server_metrics_row_index


def _metrics_timestamp() -> dict[str, str | float]:
    timestamp = make_timestamp()
    return {
        "timestamp_iso": timestamp["iso"],
        "timestamp_unix": timestamp["unix"],
    }


def _collect_memory_stats() -> dict[str, float | int | str]:
    if os.getenv("MLX_HARMONY_SERVER_COLLECT_MEMORY", "0") != "1":
        return {}
    try:
        import mlx.core as mx
    except Exception:
        return {}
    if not hasattr(mx, "metal"):
        return {}
    try:
        info = mx.metal.device_info()
    except Exception:
        return {}
    if not isinstance(info, dict):
        return {}
    stats: dict[str, float | int | str] = {}
    for key, value in info.items():
        if isinstance(value, (int, float, str)):
            stats[f"memory_{key}"] = value
    return stats


def _write_server_metrics(
    *,
    generator: TokenGenerator,
    prompt_tokens: int,
    completion_tokens: int,
    elapsed_seconds: float,
    prompt_start_delta: float | None,
    prefill_seconds: float | None,
    debug_path: Path,
) -> None:
    timing_stats = getattr(generator, "last_timing_stats", None) or {}
    max_kv_size = None
    if getattr(generator, "prompt_config", None) is not None:
        max_kv_size = getattr(generator.prompt_config, "max_kv_size", None)
    kv_len = prompt_tokens
    if max_kv_size is not None:
        try:
            kv_len = min(prompt_tokens, int(max_kv_size))
        except (TypeError, ValueError):
            kv_len = prompt_tokens
    tokens_per_second = completion_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    write_debug_metrics(
        debug_path=debug_path,
        metrics={
            "row_index": _next_server_metrics_row_index(),
            **_metrics_timestamp(),
            "prompt_tokens": prompt_tokens,
            "kv_len": kv_len,
            "completion_tokens": completion_tokens,
            "generated_tokens": completion_tokens,
            "prefill_seconds": prefill_seconds or timing_stats.get("prefill"),
            "elapsed_seconds": elapsed_seconds,
            "tokens_per_second": tokens_per_second,
            "prompt_start_to_prompt_start_seconds": prompt_start_delta,
            "max_context_tokens": getattr(generator.prompt_config, "max_context_tokens", None)
            if getattr(generator, "prompt_config", None) is not None
            else None,
            "max_kv_size": max_kv_size,
            "repetition_window": None,
            "loop_detection_mode": None,
            "prefill_start_offset": getattr(generator, "_last_prefill_start_offset", None),
            **_collect_memory_stats(),
        },
    )


def _should_log_prompt_response() -> bool:
    return os.getenv("MLX_HARMONY_SERVER_LOG_PROMPTS", DEFAULT_SERVER_LOG_PROMPTS) == "1"


def _build_server_conversation(
    *,
    request: ChatRequest,
    prompt_config: object | None,
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for msg in request.messages:
        content = msg.content
        if prompt_config and msg.role == "user":
            content = apply_placeholders(content, getattr(prompt_config, "placeholders", {}))
        messages.append({"role": msg.role, "content": content})

    has_assistant_message = any(msg.get("role") == "assistant" for msg in messages)
    if (
        prompt_config
        and getattr(prompt_config, "assistant_greeting", None)
        and not has_assistant_message
    ):
        greeting_text = apply_placeholders(
            getattr(prompt_config, "assistant_greeting", ""),
            getattr(prompt_config, "placeholders", {}),
        )
        messages.insert(0, {"role": "assistant", "content": greeting_text})

    conversation: list[dict[str, object]] = []
    parent_id: str | None = None
    for msg in messages:
        message_id = make_message_id()
        conversation.append(
            {
                "id": message_id,
                "parent_id": parent_id,
                "cache_key": message_id,
                "role": msg.get("role"),
                "content": msg.get("content"),
                "timestamp": make_timestamp(),
            }
        )
        parent_id = message_id
    return conversation


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible chat completions endpoint.
    """
    global _last_prompt_start_time
    model_path, prompt_config_path, profile_data = _resolve_profile(request)

    generator = _get_generator(model_path, prompt_config_path)
    raw_messages = [{"role": m.role, "content": m.content} for m in request.messages]
    created = int(time.time())
    response_id = f"chatcmpl-{created}"
    debug_path = Path(os.getenv("MLX_HARMONY_SERVER_DEBUG_LOG", DEFAULT_SERVER_DEBUG_LOG))

    if request.stream:

        def generate_stream():
            prompt_config = getattr(generator, "prompt_config", None)
            conversation = _build_server_conversation(
                request=request,
                prompt_config=prompt_config,
            )
            system_message = None
            prompt_conversation, prompt_token_count = truncate_conversation_for_context(
                generator=generator,
                conversation=conversation,
                system_message=system_message,
                max_context_tokens=resolve_max_context_tokens(
                    args=argparse.Namespace(max_context_tokens=None),
                    loaded_max_context_tokens=None,
                    loaded_model_path=None,
                    prompt_config=prompt_config,
                    profile_data=profile_data,
                    model_path=model_path,
                ),
                max_context_tokens_margin=getattr(prompt_config, "max_context_tokens_margin", None)
                if prompt_config
                else None,
            )
            prompt_token_ids = build_prompt_token_ids(
                generator=generator,
                conversation=prompt_conversation,
                system_message=system_message,
            )
            _ = prepare_prompt(
                generator=generator,
                conversation=prompt_conversation,
                system_message=system_message,
                debug_path=debug_path,
                debug=False,
                debug_tokens=None,
                prompt_token_ids=prompt_token_ids,
            )
            prompt_start = time.perf_counter()
            global _last_prompt_start_time
            prompt_start_delta = None
            if _last_prompt_start_time is not None:
                prompt_start_delta = prompt_start - _last_prompt_start_time
            _last_prompt_start_time = prompt_start
            generation_start = time.perf_counter()
            completion_tokens = 0
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_path, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
            collected_chunks: list[str] = []
            adapter = get_adapter(generator)
            tokens, all_generated_tokens, streamed_text_parts = adapter.stream(
                generator=generator,
                conversation=prompt_conversation,
                system_message=system_message,
                prompt_token_ids=prompt_token_ids,
                hyperparameters={
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "top_p": request.top_p,
                    "min_p": request.min_p,
                    "top_k": request.top_k,
                    "repetition_penalty": request.repetition_penalty,
                    "repetition_context_size": request.repetition_context_size,
                },
                seed=None,
                on_text=lambda _text: None,
            )
            for token_id in tokens:
                completion_tokens += 1
                text = _decode_tokens(generator, [int(token_id)])
                if _should_log_prompt_response():
                    collected_chunks.append(text)
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_path,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            generation_end = time.perf_counter()
            elapsed = generation_end - generation_start
            _write_server_metrics(
                generator=generator,
                prompt_tokens=prompt_token_count,
                completion_tokens=completion_tokens,
                elapsed_seconds=elapsed,
                prompt_start_delta=prompt_start_delta,
                prefill_seconds=None,
                debug_path=Path(debug_path),
            )
            write_debug_tokens(
                debug_path=Path(debug_path),
                token_ids=all_generated_tokens,
                decode_tokens=generator.encoding.decode if generator.encoding else None,
                label="response",
                mode="off",
            )
            raw_response = adapter.decode_raw(
                generator=generator,
                prompt_token_ids=prompt_token_ids,
                all_generated_tokens=all_generated_tokens,
            )
            parsed = adapter.parse(
                generator=generator,
                tokens=tokens,
                streamed_text_parts=streamed_text_parts,
                assistant_name=get_assistant_name(prompt_config),
                thinking_limit=get_truncate_limits(prompt_config)[0],
                response_limit=get_truncate_limits(prompt_config)[1],
                render_markdown=False,
                debug=False,
                display_assistant=lambda *_args, **_kwargs: None,
                display_thinking=lambda *_args, **_kwargs: None,
                truncate_text=lambda text, _limit: text,
                suppress_display=True,
            )
            if _should_log_prompt_response():
                try:
                    write_debug_response(
                        debug_path=Path(debug_path),
                        raw_response=raw_response,
                        cleaned_response=parsed.assistant_text or raw_response,
                        show_console=False,
                    )
                except Exception as exc:
                    logger.warning("Failed to write server debug response: %s", exc)
            write_debug_token_texts(
                debug_path=Path(debug_path),
                token_ids=all_generated_tokens,
                decode_token=generator.encoding.decode if generator.encoding else generator.tokenizer.decode,
                label="response",
                mode="off",
            )
            finish_reason = generator.last_finish_reason
            if not isinstance(finish_reason, str) or not finish_reason:
                finish_reason = "stop"
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_path, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    prompt_config = getattr(generator, "prompt_config", None)
    assistant_name = get_assistant_name(prompt_config)
    thinking_limit, response_limit = get_truncate_limits(prompt_config)
    render_markdown = False
    tools: list[object] = []

    conversation = _build_server_conversation(
        request=request,
        prompt_config=prompt_config,
    )
    messages = [{"role": msg.get("role"), "content": msg.get("content")} for msg in conversation]

    args = argparse.Namespace(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        min_p=request.min_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        repetition_context_size=request.repetition_context_size,
        loop_detection=None,
        max_context_tokens=None,
        seed=None,
        reseed_each_turn=None,
    )
    hyperparameters = build_hyperparameters(
        args,
        loaded_hyperparameters={},
        prompt_config=prompt_config,
        is_harmony=bool(getattr(generator, "is_gpt_oss", False) and getattr(generator, "use_harmony", False)),
    )
    last_saved_hyperparameters: dict[str, float | int | bool | str] = {}
    max_context_tokens = resolve_max_context_tokens(
        args=args,
        loaded_max_context_tokens=None,
        loaded_model_path=None,
        prompt_config=prompt_config,
        profile_data=profile_data,
        model_path=model_path,
    )

    last_user_text = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_text = str(msg.get("content") or "")
            break

    result = run_chat_turn(
        generator=generator,
        conversation=conversation,
        hyperparameters=hyperparameters,
        last_saved_hyperparameters=last_saved_hyperparameters,
        assistant_name=assistant_name,
        thinking_limit=thinking_limit,
        response_limit=response_limit,
        render_markdown=render_markdown,
        debug=False,
        debug_path=debug_path,
        debug_tokens=None,
        enable_artifacts=True,
        max_context_tokens=max_context_tokens,
        max_tool_iterations=10,
        max_resume_attempts=2,
        tools=tools,
        last_user_text=last_user_text,
        make_message_id=make_message_id,
        make_timestamp=make_timestamp,
        display_assistant=lambda *_args, **_kwargs: None,
        display_thinking=lambda *_args, **_kwargs: None,
        truncate_text=lambda text, _limit: text,
        collect_memory_stats=_collect_memory_stats,
        write_debug_metrics=write_debug_metrics,
        write_debug_response=write_debug_response,
        write_debug_token_texts=write_debug_token_texts,
        write_debug_tokens=write_debug_tokens,
        last_prompt_start_time=_last_prompt_start_time,
        generation_index=0,
    )
    _last_prompt_start_time = result.last_prompt_start_time

    assistant_text = ""
    analysis_text: str | None = None
    for msg in reversed(conversation):
        if msg.get("role") == "assistant":
            assistant_text = str(msg.get("content") or "")
            if request.return_analysis and msg.get("analysis"):
                analysis_text = str(msg.get("analysis"))
            break

    prompt_tokens = result.prompt_tokens or 0
    completion_tokens = result.completion_tokens or 0
    total_tokens = prompt_tokens + completion_tokens
    finish_reason = generator.last_finish_reason
    if not isinstance(finish_reason, str) or not finish_reason:
        finish_reason = "stop"
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_path,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                "content": assistant_text,
                **({"analysis": analysis_text} if analysis_text else {}),
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


@app.get("/v1/models")
async def list_models():
    models_dir = os.getenv("MLX_HARMONY_MODELS_DIR", DEFAULT_MODELS_DIR)
    data = []
    base = Path(models_dir)
    if base.exists():
        for entry in sorted(base.iterdir()):
            if entry.is_dir():
                data.append(
                    {
                        "id": str(entry),
                        "object": "model",
                        "created": 0,
                        "owned_by": "local",
                    }
                )
    if not data:
        profiles_path = os.getenv("MLX_HARMONY_PROFILES_FILE", DEFAULT_PROFILES_FILE)
        if os.path.exists(profiles_path):
            try:
                profiles = load_profiles(profiles_path)
            except Exception as exc:
                logger.warning("Failed to load profiles from %s: %s", profiles_path, exc)
                profiles = {}
            for name, profile in profiles.items():
                model_id = profile.get("model", name)
                data.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "local",
                    }
                )
    return {"object": "list", "data": data}


@app.get("/v1/health")
async def health_check() -> dict[str, str | bool | int | None]:
    model_loaded = _generator is not None
    return {
        "object": "health",
        "status": "ok",
        "model_loaded": model_loaded,
        "model_path": _loaded_model_path,
        "prompt_config_path": _generator_prompt_config_path,
        "loaded_at_unix": _loaded_at_unix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MLX Harmony API server.")
    parser.add_argument(
        "--host",
        default=os.getenv("MLX_HARMONY_HOST", "0.0.0.0"),
        help="Host interface to bind the server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MLX_HARMONY_PORT", "8000")),
        help="Port to bind the server.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("MLX_HARMONY_LOG_LEVEL", "info"),
        help="Uvicorn log level (debug, info, warning, error).",
    )
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MLX_HARMONY_RELOAD", "false").lower() == "true",
        help="Enable auto-reload on code changes (development only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("MLX_HARMONY_WORKERS", "1")),
        help="Number of worker processes (use 1 for in-process model cache).",
    )
    parser.add_argument(
        "--profiles-file",
        default=os.getenv("MLX_HARMONY_PROFILES_FILE", DEFAULT_PROFILES_FILE),
        help="Profiles file path used for model selection.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MLX_HARMONY_MODEL_PATH", None),
        help="Model path for preload (optional).",
    )
    parser.add_argument(
        "--profile",
        default=os.getenv("MLX_HARMONY_PROFILE", None),
        help="Profile name for preload (optional).",
    )
    parser.add_argument(
        "--prompt-config",
        default=os.getenv("MLX_HARMONY_PROMPT_CONFIG", None),
        help="Prompt config path for preload (optional).",
    )
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("MLX_HARMONY_PRELOAD", "false").lower() == "true",
        help="Preload model at startup (default: false).",
    )
    args: argparse.Namespace = parser.parse_args()

    os.environ["MLX_HARMONY_PROFILES_FILE"] = args.profiles_file
    if args.preload:
        try:
            model_path, prompt_config_path = _resolve_profile_from_args(
                model=args.model,
                profile=args.profile,
                prompt_config=args.prompt_config,
                profiles_file=args.profiles_file,
            )
        except RuntimeError as exc:
            logger.error("Failed to preload model: %s", exc)
        else:
            if model_path:
                _get_generator(model_path, prompt_config_path)
            else:
                logger.warning(
                    "Preload requested but no model/profile provided; skipping preload."
                )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
