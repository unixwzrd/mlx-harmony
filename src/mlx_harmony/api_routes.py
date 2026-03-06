"""Reusable API route builders for HTTP endpoint registration."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from fastapi.responses import JSONResponse

from mlx_harmony.api_contract import (
    build_health_response,
    build_models_list_response,
    build_openai_error_response,
)


def build_not_implemented_response(endpoint: str) -> JSONResponse:
    """Build a standardized 501 payload for unimplemented endpoints.

    Args:
        endpoint: Endpoint path identifier.

    Returns:
        OpenAI-style `501` JSON response.
    """

    return JSONResponse(
        status_code=501,
        content=build_openai_error_response(
            message=f"Endpoint '{endpoint}' is not implemented yet.",
            error_type="not_implemented_error",
            code="not_implemented",
        ),
    )


def build_invalid_request_response(message: str, *, param: str | None = None) -> JSONResponse:
    """Build a standardized 400 invalid_request_error payload.

    Args:
        message: Human-readable validation or request resolution message.
        param: Optional request parameter name associated with the error.

    Returns:
        OpenAI-style `400` JSON response.
    """

    return JSONResponse(
        status_code=400,
        content=build_openai_error_response(
            message=message,
            error_type="invalid_request_error",
            param=param,
            code=None,
        ),
    )


def _build_not_implemented_handler(endpoint: str) -> Callable[[], Awaitable[JSONResponse]]:
    """Create one async placeholder handler for a specific endpoint path.

    Args:
        endpoint: Endpoint path.

    Returns:
        Async endpoint handler returning a `501` payload.
    """

    async def _handler() -> JSONResponse:
        return build_not_implemented_response(endpoint)

    return _handler


def register_placeholder_post_endpoints(
    *,
    app: FastAPI,
    endpoints: Iterable[str],
) -> None:
    """Register OpenAI-compatible placeholder POST endpoints.

    Args:
        app: FastAPI app instance.
        endpoints: Endpoint paths to register.
    """

    for endpoint in endpoints:
        route_name = endpoint.strip("/").replace("/", "_").replace("-", "_") + "_placeholder"
        app.add_api_route(
            endpoint,
            _build_not_implemented_handler(endpoint),
            methods=["POST"],
            name=route_name,
        )


def register_chat_completions_route(
    *,
    app: FastAPI,
    handler: Callable[[], Awaitable[Any]],
) -> None:
    """Register `/v1/chat/completions` route with a caller-provided handler.

    Args:
        app: FastAPI app instance.
        handler: Async handler that executes one chat-completions request.
    """

    app.add_api_route(
        "/v1/chat/completions",
        handler,
        methods=["POST"],
        name="chat_completions",
    )


def register_models_route(
    *,
    app: FastAPI,
    get_model_ids: Callable[[], list[str]],
) -> None:
    """Register `/v1/models` route with caller-provided model id lookup.

    Args:
        app: FastAPI app instance.
        get_model_ids: Callback returning model identifiers.
    """

    async def _list_models() -> dict[str, Any]:
        return build_models_list_response(model_ids=get_model_ids())

    app.add_api_route("/v1/models", _list_models, methods=["GET"], name="list_models")


def register_health_route(
    *,
    app: FastAPI,
    get_model_loaded: Callable[[], bool],
    get_model_path: Callable[[], str | None],
    get_prompt_config_path: Callable[[], str | None],
    get_loaded_at_unix: Callable[[], int | None],
) -> None:
    """Register `/v1/health` route with caller-provided state readers.

    Args:
        app: FastAPI app instance.
        get_model_loaded: Callback returning model-loaded state.
        get_model_path: Callback returning loaded model path.
        get_prompt_config_path: Callback returning loaded prompt config path.
        get_loaded_at_unix: Callback returning load timestamp.
    """

    async def _health_check() -> dict[str, str | bool | int | None]:
        return build_health_response(
            model_loaded=get_model_loaded(),
            model_path=get_model_path(),
            prompt_config_path=get_prompt_config_path(),
            loaded_at_unix=get_loaded_at_unix(),
        )

    app.add_api_route("/v1/health", _health_check, methods=["GET"], name="health_check")


def register_request_validation_handler(
    *,
    app: FastAPI,
    invalid_request_response_builder: Callable[[str], JSONResponse]
    | Callable[..., JSONResponse],
) -> None:
    """Register a shared FastAPI request-validation error adapter.

    Args:
        app: FastAPI app instance.
        invalid_request_response_builder: Callable that returns OpenAI-style
            invalid request responses.
    """

    @app.exception_handler(FastAPIRequestValidationError)
    async def _fastapi_validation_error_handler(
        _request: Any, exc: FastAPIRequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        if not errors:
            return invalid_request_response_builder("Invalid request payload.")
        first = errors[0]
        message = str(first.get("msg") or "Invalid request payload.")
        loc = first.get("loc")
        param: str | None = None
        if isinstance(loc, (list, tuple)) and loc:
            parts = [str(part) for part in loc if str(part) != "body"]
            if parts:
                param = ".".join(parts)
        return invalid_request_response_builder(message, param=param)
