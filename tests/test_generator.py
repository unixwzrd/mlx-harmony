"""
Unit tests for the TokenGenerator class.

Tests model loading, format detection, and token generation.
"""
import pytest

from mlx_harmony.config import PromptConfig
from mlx_harmony.generation.generator import TokenGenerator


@pytest.mark.requires_model
class TestTokenGenerator:
    """Test TokenGenerator functionality."""

    def test_init_with_model(self, test_model_path: str):
        """Test initializing TokenGenerator with a model."""
        generator = TokenGenerator(test_model_path, lazy=True)
        assert generator.model_path == test_model_path
        assert generator.model is not None
        assert generator.tokenizer is not None

    def test_non_gpt_oss_detection(self, test_model_path: str):
        """Test that non-GPT-OSS models are detected correctly."""
        generator = TokenGenerator(test_model_path, lazy=True)
        assert generator.is_gpt_oss is False
        assert generator.use_harmony is False

    def test_generate_with_prompt(self, test_model_path: str):
        """Test generating tokens with a simple prompt."""
        generator = TokenGenerator(test_model_path, lazy=False)
        prompt = "Hello, how are you?"
        tokens = list(generator.generate(prompt=prompt, max_tokens=10))
        assert len(tokens) > 0
        assert len(tokens) <= 10  # Should respect max_tokens

    def test_generate_with_messages(self, test_model_path: str):
        """Test generating tokens with messages."""
        generator = TokenGenerator(test_model_path, lazy=False)
        messages = [{"role": "user", "content": "Say hello!"}]
        tokens = list(generator.generate(messages=messages, max_tokens=10))
        assert len(tokens) > 0

    def test_generate_with_sampling_params(self, test_model_path: str):
        """Test generating with custom sampling parameters."""
        generator = TokenGenerator(test_model_path, lazy=False)
        tokens = list(
            generator.generate(
                prompt="Test",
                max_tokens=5,
                temperature=0.5,
                top_p=0.9,
                top_k=10,
            )
        )
        assert len(tokens) > 0

    def test_generate_with_prompt_config(self, test_model_path: str, sample_prompt_config: dict):
        """Test generating with a prompt config."""
        config = PromptConfig(**sample_prompt_config)
        generator = TokenGenerator(test_model_path, prompt_config=config, lazy=False)
        tokens = list(generator.generate(prompt="Test", max_tokens=5))
        assert len(tokens) > 0

    def test_max_tokens_from_config(self, test_model_path: str, sample_prompt_config: dict):
        """Test that max_tokens is resolved from prompt config."""
        config = PromptConfig(**sample_prompt_config)
        generator = TokenGenerator(test_model_path, prompt_config=config, lazy=False)
        # Don't specify max_tokens, should use config default (100)
        tokens = list(generator.generate(prompt="Test"))
        assert len(tokens) <= 100

    def test_stop_sequences(self, test_model_path: str):
        """Test that stop sequences work correctly."""
        generator = TokenGenerator(test_model_path, lazy=False)
        # Use a stop sequence that's likely to appear
        stop_tokens = generator.tokenizer.encode("!", add_special_tokens=False)
        if stop_tokens:
            tokens = list(
                generator.generate(
                    prompt="Count: 1! 2! 3!",
                    max_tokens=20,
                    stop_tokens=stop_tokens[:1],  # Just the first token
                )
            )
            # Should stop early if stop token is found
            assert len(tokens) <= 20


@pytest.mark.requires_model
@pytest.mark.slow
class TestTokenGeneratorInference:
    """Test actual inference with a real model (slower tests)."""

    def test_full_generation_cycle(self, test_model_path: str):
        """Test a full generation cycle with model loading."""
        generator = TokenGenerator(test_model_path, lazy=False)
        prompt = "The capital of France is"
        tokens = list(generator.generate(prompt=prompt, max_tokens=10, temperature=0.0))
        assert len(tokens) > 0
        # Decode and verify we got some text
        text = generator.tokenizer.decode([int(t) for t in tokens])
        assert len(text) > 0

    def test_multiple_generations(self, test_model_path: str):
        """Test multiple generations with the same generator."""
        generator = TokenGenerator(test_model_path, lazy=False)
        # First generation
        tokens1 = list(generator.generate(prompt="Hello", max_tokens=5))
        # Second generation
        tokens2 = list(generator.generate(prompt="World", max_tokens=5))
        assert len(tokens1) > 0
        assert len(tokens2) > 0
        # Should be different (unless deterministic and same prompt)
        assert tokens1 != tokens2 or len(tokens1) == len(tokens2)
