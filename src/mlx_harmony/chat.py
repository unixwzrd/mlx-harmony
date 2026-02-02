from __future__ import annotations

from mlx_harmony.chat_bootstrap import bootstrap_chat
from mlx_harmony.chat_frontend import run_cli_frontend
from mlx_harmony.chat_io import load_conversation, save_conversation, try_save_conversation

__all__ = ["load_conversation", "save_conversation", "try_save_conversation", "main"]


def main() -> None:
    bootstrap = bootstrap_chat()
    run_cli_frontend(
        args=bootstrap.args,
        context=bootstrap.context,
        conversation=bootstrap.conversation,
        model_path=bootstrap.model_path,
        prompt_config_path=bootstrap.prompt_config_path,
        loaded_hyperparameters=bootstrap.loaded_hyperparameters,
        loaded_max_context_tokens=bootstrap.loaded_max_context_tokens,
        loaded_model_path=bootstrap.loaded_model_path,
        loaded_chat_id=bootstrap.loaded_chat_id,
    )


if __name__ == "__main__":
    main()
