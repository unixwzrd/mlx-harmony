# Native Tokenizer Implementation

**Created**: 2026-01-09
**Updated**: 2026-01-25

## Status

Implemented and in use. The loader always uses the native tokenizer for `tokenizer.json`
ByteLevel BPE models. There is no transformers fallback.

## Current Implementation

The native tokenizer lives in [src/mlx_harmony/tokenizer_native.py](../src/mlx_harmony/tokenizer_native.py) and provides:

- ByteLevel BPE encoding/decoding (GPT-2/GPT-OSS style)
- Streaming detokenization
- Simple chat template rendering via string replacement (no Jinja2)
- Special-token handling for Harmony formatting tokens

Model loading uses the native tokenizer unconditionally in
[src/mlx_harmony/loader.py](../src/mlx_harmony/loader.py).

## Supported Tokenizers

- `tokenizer.json` with `"model": { "type": "BPE" }`

## Known Gaps

- SentencePiece (`tokenizer.model`) is not supported yet.
- TikToken models are not supported yet.
- Jinja2 chat templates are not supported; only simple string replacement is used.

## Future Enhancements

If we need broader tokenizer compatibility, add a router that detects tokenizer type
and supports the following:

1. SentencePiece via `sentencepiece.SentencePieceProcessor`
2. TikToken for GPT-style models that ship `.tiktoken` assets
3. BPE remains the default

## References

- `mlx-examples/llms/llama/llama.py` - SentencePiece example
- `mlx-examples/whisper/mlx_whisper/tokenizer.py` - TikToken example
- `mlx-examples/clip/tokenizer.py` - Native BPE example
- `mlx-examples/flux/flux/tokenizers.py` - CLIP and T5 tokenizers
