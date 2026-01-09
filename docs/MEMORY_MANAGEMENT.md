# Memory Management Guide

This guide explains how memory management works in `mlx-harmony`, including wired memory (mlock) and considerations for loading multiple models.

## Table of Contents

- [Overview](#overview)
- [Wired Memory (mlock)](#wired-memory-mlock)
- [Multiple Models](#multiple-models)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

`mlx-harmony` provides wired memory (mlock) support to optimize inference performance:

- **Wired memory (mlock)**: Locks model weights in physical RAM to prevent swapping, improving inference performance.

This feature is optional and can be enabled via configuration or CLI arguments.

## Wired Memory (mlock)

### What is Wired Memory?

Wired memory (also called "mlock" or "resident memory") keeps memory pages in physical RAM and prevents the operating system from swapping them to disk. This is particularly important for large language models because:

- **Performance**: Swapping model weights to disk causes severe performance degradation (100x+ slower)
- **Consistency**: Keeps inference latency predictable
- **Reliability**: Prevents out-of-memory errors during inference

### How It Works

When `mlock: true` is enabled, `mlx-harmony`:

1. **Sets the wired limit** before loading the model to `max_recommended_working_set_size` (the maximum allowed by macOS)
2. **Wires buffers as they're allocated** during model loading
3. **Keeps model weights in wired memory** for the lifetime of the model

The wired limit is a **capacity limit**, not an allocation. MLX will wire buffers up to this limit as they're allocated, but won't reserve unused memory.

#### Technical Details

- **Platform**: Requires macOS 15.0+ with Metal backend
- **Implementation**: Uses MLX's `set_wired_limit()` function, which manages a Metal ResidencySet
- **Limit**: Set to `max_recommended_working_set_size` from `mx.metal.device_info()`
- **Behavior**: Buffers are automatically added to the wired set when allocated, if they fit within the capacity limit

### When to Enable

Enable wired memory when:
- ✅ Running inference on large models (>10GB)
- ✅ Need consistent, low-latency inference
- ✅ Model fits in available RAM
- ✅ macOS 15.0+ with Metal GPU

Disable wired memory when:
- ❌ Running on non-macOS systems
- ❌ System has limited RAM
- ❌ Loading multiple very large models simultaneously (see [Multiple Models](#multiple-models))

### System Requirements

The wired limit is bounded by the macOS system wired limit, which defaults to a conservative value. For large models, you may need to increase it:

```bash
sudo sysctl iogpu.wired_limit_mb=<size_in_megabytes>
```

Where `<size_in_megabytes>` should be:
- Larger than your model size (in MB)
- Smaller than your total system RAM (in MB)
- Typically: model_size_mb * 1.1 to 1.2

**Example**: For a 20GB model on a 64GB Mac:
```bash
sudo sysctl iogpu.wired_limit_mb=22000
```

To make this permanent, add it to `/etc/sysctl.conf`:
```
iogpu.wired_limit_mb=22000
```

## Multiple Models

### Is It Safe?

**Yes, but with caveats.** Loading multiple models with `mlock: true` is supported, but you need to be aware of how the wired limit works:

- **Global limit**: The wired limit is **global** across all MLX operations
- **Shared capacity**: All models share the same wired memory capacity
- **Competition**: Models compete for wired memory slots as they load

### Best Practices

1. **Check total size**: Sum the sizes of all models you plan to load simultaneously
   ```bash
   # Check model sizes
   du -sh ~/models/model1 ~/models/model2
   ```

2. **Ensure sufficient capacity**: Total model sizes should fit within `max_recommended_working_set_size`
   - If models total 40GB but max is 32GB, some buffers will be unwired
   - Consider increasing system wired limit if needed

3. **Monitor memory**: Watch `Activity Monitor` or use `vm_stat` to see wired memory usage
   ```bash
   vm_stat | grep "Pages wired down"
   ```

4. **Consider alternatives**:
   - Load models sequentially (unload one before loading another)
   - Use smaller models when loading multiple
   - Disable `mlock` for less-critical models

### Example Scenarios

#### Scenario 1: Two 10GB Models (20GB total)
- System max: 32GB
- **Result**: ✅ Both models can fit in wired memory
- **Action**: Enable `mlock: true` for both

#### Scenario 2: One 30GB Model
- System max: 32GB
- **Result**: ✅ Model fits with room for activations
- **Action**: Enable `mlock: true`

#### Scenario 3: Three 15GB Models (45GB total)
- System max: 32GB
- **Result**: ❌ Total exceeds capacity, some buffers will be unwired
- **Action**: 
  - Increase system wired limit if you have RAM
  - Or disable `mlock` for one or more models
  - Or load models sequentially

## Configuration

### Prompt Config JSON

Add `mlock` to your prompt config JSON:

```json
{
  "system_model_identity": "You are {assistant}.",
  "temperature": 0.8,
  "max_tokens": 1024,
  "mlock": false
}
```

### CLI Arguments

Override config values via command line:

```bash
# Enable mlock via CLI
mlx-harmony-chat --model ~/models/my-model --mlock
```

### Priority

Parameter priority (highest to lowest):

1. **CLI arguments** (`--mlock`)
2. **Prompt config JSON** (`mlock` field)
3. **Default values** (`mlock: false`)

### Profile Configuration

You can set `mlock` in the referenced prompt config within a profile:

```json
{
  "model_path": "~/models/gpt-oss-20b",
  "prompt_config_path": "configs/my-config.json"
}
```

Where `configs/my-config.json` contains:

```json
{
  "mlock": true
}
```

## Troubleshooting

### "Model weights not staying in wired memory"

**Symptoms**: Memory utilization jumps during inference, model not in "Wired" memory in Activity Monitor.

**Solutions**:
1. Ensure `mlock: true` is set in config or `--mlock` flag is used
2. Check macOS version: Requires macOS 15.0+
3. Verify Metal backend: `python -c "import mlx.core as mx; print(mx.metal.is_available())"`
4. Check system wired limit: May need to increase with `sysctl iogpu.wired_limit_mb`
5. Ensure model size < system wired limit

### "Failed to set wired limit" warning

**Causes**:
- Not on macOS
- macOS version < 15.0
- Metal backend unavailable
- Model size exceeds system limit

**Solutions**:
- Upgrade to macOS 15.0+ if possible
- Check Metal availability
- Increase system wired limit if model is too large

### High memory usage with multiple models

**Symptom**: Loading multiple models causes memory pressure or OOM errors.

**Solutions**:
1. Calculate total model sizes and ensure they fit in RAM
2. Reduce number of models loaded simultaneously
3. Disable `mlock` for less-critical models
4. Consider model quantization to reduce sizes

## Performance Tips

1. **Development**: Disable `mlock` to reduce memory pressure during testing
2. **Production**: Enable `mlock` for consistent inference latency
3. **Large models**: Always check system wired limit before enabling `mlock`
4. **Multiple models**: Calculate total sizes and monitor wired memory usage

## References

- [MLX Memory Management Documentation](https://ml-explore.github.io/mlx/build/html/python/memory_management.html)
- [MLX-LM README - Large Models](https://github.com/ml-explore/mlx-lm/tree/main#large-models)
- [macOS Metal Residency Sets](https://developer.apple.com/documentation/metal/mtlresidencyset)

## Summary

- **Wired memory (mlock)**: Keeps model weights in RAM, prevents swapping
  - ✅ Improves inference performance significantly
  - ✅ Requires macOS 15.0+ with Metal
  - ✅ Safe for single models that fit in RAM
  - ⚠️ Use caution with multiple large models

- **Multiple models**: Supported but watch total memory usage
  - ✅ Safe if total sizes < system wired limit
  - ⚠️ Models compete for wired memory slots
  - 💡 Consider sequential loading for very large models

---

[← Back to README](../README.md)
