# Profiling Scripts

## Profile Running Chat (Real-World Usage) ⭐ Recommended

Use `profile_chat.py` to profile the actual `mlx-harmony-chat` command as it runs. This captures **real-world performance** including:

- Model loading
- Token generation
- Message parsing
- Full chat loop interactions
- User input handling

This is more useful than `profile_startup.py` because it shows performance during actual usage.

### Quick Start

```bash
# Profile chat with one interaction (type 'q' to quit and finish profiling)
python scripts/profile_chat.py \
  --model models/huizimao-gpt-oss-20b-uncensored-mxfp4-q8-hi-mlx \
  --prompt-config configs/Mia.json

# Profile with specific chat arguments
python scripts/profile_chat.py \
  --model models/my-model \
  --temperature 0.8 \
  --max-tokens 100 \
  --profile-output chat_profile.stats \
  --graph chat_profile.svg

# Text-only report (faster, no graphviz needed)
python scripts/profile_chat.py \
  --model models/my-model \
  --prompt-config configs/Mia.json \
  --text-only
```

**Note**: This will start the actual chat interface. Interact with it normally (ask questions, have a conversation), then type 'q' to quit. The profiling data will be saved when you exit. All operations during the chat session are profiled.

### Output Files

All profiling output files are saved to the `stats/` directory by default:

- `stats/profile_chat.stats` - cProfile stats file (can be viewed with `python -m pstats`)
- `stats/profile_chat.stats.txt` - Text report with top functions
- `stats/profile_chat.svg` - Graphviz call graph (if `gprof2dot` + Graphviz `dot` are installed)

### Viewing Results

Same as `profile_startup.py` - see the "Viewing Results" section below.

---

## Profile Startup Performance

Use `profile_startup.py` to identify performance bottlenecks during model loading and initialization only (doesn't run the full chat loop).

### Quick Start

```bash
# Profile TokenGenerator initialization (model loading)
python scripts/profile_startup.py \
  --model models/huizimao-gpt-oss-20b-uncensored-mxfp4-q8-hi-mlx \
  --prompt-config configs/Mia.json

# Generate both text and graphviz visualization
python scripts/profile_startup.py \
  --model <model_path> \
  --output profile.stats \
  --graph profile.svg

# Text-only report (faster)
python scripts/profile_startup.py \
  --model <model_path> \
  --text-only
```

### Output Files

All profiling output files are saved to the `stats/` directory by default:

- `stats/profile_startup.stats` - cProfile stats file (can be viewed with `python -m pstats`)
- `stats/profile_startup.stats.txt` - Text report with top functions
- `stats/profile_startup.svg` - Graphviz call graph (if `gprof2dot` + Graphviz `dot` are installed)

### Viewing Results

**Text Report:**

```bash
# View the text report
cat profile_startup.stats.txt

# Or use pstats interactively
python -m pstats profile_startup.stats
```

**Graphviz Visualization:**

```bash
# Install gprof2dot (if not already installed)
pip install gprof2dot

# Install Graphviz (provides the `dot` binary)
brew install graphviz  # macOS

# Generate SVG
python scripts/profile_startup.py --model <model_path> --graph stats/profile.svg

# View on macOS
open stats/profile.svg

# View on Linux
xdg-open stats/profile.svg
```

**Interactive Visualization (SnakeViz):**

```bash
# Install snakeviz
pip install snakeviz

# View stats interactively in browser
snakeviz stats/profile_startup.stats
```

### Understanding the Output

The profiling report shows:

- **Cumulative time**: Total time including subcalls
- **Total time**: Time spent in the function itself (excluding subcalls)

Look for:

- Functions with high cumulative time that call many subfunctions
- Functions with high total time that are doing expensive work
- Model loading operations (`mlx_lm.load`)
- Tokenizer initialization
- Harmony encoding loading

### Example Output

```text
PROFILING REPORT (Top 50 functions by cumulative time)
============================================================
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000   45.234   45.234 generator.py:33(__init__)
        1    0.000    0.000   44.891   44.891 mlx_lm/__init__.py:123(load)
        1    0.000    0.000   38.234   38.234 mlx_lm/models/llama.py:45(load_model)
    ...
```

### Tips

1. **Model Loading is Usually the Bottleneck**: Most startup time is spent loading model weights from disk. This is expected and hard to optimize.

2. **Check for Redundant Imports**: Look for modules being imported multiple times.

3. **Lazy Loading**: Consider using `lazy=True` in `TokenGenerator` if you want to defer model loading.

4. **Profile Specific Operations**: Modify the script to profile specific parts of the startup if needed.
