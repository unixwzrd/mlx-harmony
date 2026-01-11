# Native Tokenizer Implementation Proposal

**Status:** Proposal / Future Improvement  
**Priority:** Medium  
**Last Updated:** 2026-01-09

## Problem

Currently, `mlx-harmony` (via `mlx-lm`) uses `transformers.AutoTokenizer` for tokenizer loading, which imports PyTorch when available. While PyTorch is only used during tokenizer initialization (not during MLX inference), this creates an unnecessary dependency and startup overhead.

## Solution

Implement native tokenizer loading similar to how `mlx-examples` handles it:

1. **SentencePiece models** (Llama, Mistral, Qwen, etc.): Use `sentencepiece.SentencePieceProcessor` directly
2. **GPT-style models**: Use `tiktoken` for models that support it
3. **BPE models**: Implement native BPE tokenizer (like CLIP example)
4. **Other models**: Fall back to transformers only when necessary

## Benefits

- **No PyTorch dependency**: Eliminates transitive PyTorch import
- **Faster startup**: Native tokenizers load faster
- **Smaller footprint**: Fewer dependencies to install
- **Better MLX alignment**: Follows the philosophy of mlx-examples

## Implementation Plan

### Phase 1: Detection & Routing

1. Detect tokenizer type from model files:
   - Check for `tokenizer.model` (SentencePiece)
   - Check for `tokenizer.json` with decoder type
   - Check for `tokenizer_config.json` for model type hints

2. Create tokenizer loader router:
   ```python
   def load_tokenizer_native(model_path: Path) -> Tokenizer:
       if (model_path / "tokenizer.model").exists():
           return load_sentencepiece_tokenizer(model_path)
       elif supports_tiktoken(model_path):
           return load_tiktoken_tokenizer(model_path)
       # Fall back to transformers for complex cases
       return load_transformers_tokenizer(model_path)
   ```

### Phase 2: Native Implementations

1. **SentencePiece tokenizer** (high priority - most common):
   - Use `sentencepiece.SentencePieceProcessor` directly
   - Wrap in `TokenizerWrapper` for compatibility
   - Reference: `mlx-examples/llms/llama/llama.py`

2. **TikToken tokenizer** (medium priority):
   - Use `tiktoken` library for GPT-style models
   - Reference: `mlx-examples/whisper/mlx_whisper/tokenizer.py`

3. **BPE tokenizer** (lower priority):
   - Implement native BPE like CLIP example
   - Reference: `mlx-examples/clip/tokenizer.py`

### Phase 3: Chat Template Support

- Parse `tokenizer_config.json` for chat template
- Support Jinja2 templates natively (no transformers dependency)
- Maintain compatibility with existing chat template system

### Phase 4: Streaming Detokenizers

- Keep existing streaming detokenizer implementations from `mlx-lm`
- These already work natively (read from `tokenizer.json`)
- No changes needed here

## References

- `mlx-examples/llms/llama/llama.py` - SentencePiece example
- `mlx-examples/whisper/mlx_whisper/tokenizer.py` - TikToken example
- `mlx-examples/clip/tokenizer.py` - Native BPE example
- `mlx-examples/flux/flux/tokenizers.py` - CLIP and T5 tokenizers

## Migration Strategy

1. **Add native tokenizer loader** as optional alternative
2. **Feature flag** to switch between native/transformers loading
3. **Test compatibility** with existing models
4. **Make native default** once stable
5. **Remove transformers dependency** eventually (or keep as fallback)

## Dependencies

- `sentencepiece` - For SentencePiece models (already used by mlx-examples)
- `tiktoken` - For GPT-style models (optional, only for supported models)
- `jinja2` - For chat template rendering (if not already available)

## Notes

- This is a **breaking change** for internal APIs but not for user-facing APIs
- Need to maintain backward compatibility during transition
- Consider upstreaming improvements to `mlx-lm` if successful
