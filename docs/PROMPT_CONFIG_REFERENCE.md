# Prompt Config Reference

This document explains all parameters available in the prompt configuration JSON file (`--prompt-config`).

- [Prompt Config Reference](#prompt-config-reference)
  - [Harmony-Specific Parameters (GPT-OSS Models Only)](#harmony-specific-parameters-gpt-oss-models-only)
    - [`system_model_identity` (string, optional)](#system_model_identity-string-optional)
    - [`reasoning_effort` (string, optional)](#reasoning_effort-string-optional)
    - [`conversation_start_date` (string, optional)](#conversation_start_date-string-optional)
    - [`knowledge_cutoff` (string, optional)](#knowledge_cutoff-string-optional)
    - [`developer_instructions` (string, optional)](#developer_instructions-string-optional)
    - [`assistant_greeting` (string, optional)](#assistant_greeting-string-optional)
    - [`example_dialogues` (array of arrays, optional)](#example_dialogues-array-of-arrays-optional)
  - [Placeholders](#placeholders)
    - [`placeholders` (object, optional)](#placeholders-object-optional)
    - [Built-in Placeholders](#built-in-placeholders)
      - [Date Placeholders](#date-placeholders)
      - [Time Placeholders](#time-placeholders)
    - [User-defined Placeholders](#user-defined-placeholders)
    - [Placeholder Guide Summary](#placeholder-guide-summary)
  - [Sampling Parameters](#sampling-parameters)
    - [`max_tokens` (integer, optional)](#max_tokens-integer-optional)
    - [`temperature` (float, optional)](#temperature-float-optional)
    - [`top_p` (float, optional)](#top_p-float-optional)
    - [`min_p` (float, optional)](#min_p-float-optional)
    - [`top_k` (integer, optional)](#top_k-integer-optional)
    - [`min_tokens_to_keep` (integer, optional)](#min_tokens_to_keep-integer-optional)
    - [`repetition_penalty` (float, optional)](#repetition_penalty-float-optional)
    - [`repetition_context_size` (integer, optional)](#repetition_context_size-integer-optional)
    - [`xtc_probability` (float, optional)](#xtc_probability-float-optional)
    - [`xtc_threshold` (float, optional)](#xtc_threshold-float-optional)
    - [`xtc_special_tokens` (array of integers, optional)](#xtc_special_tokens-array-of-integers-optional)
  - [Model Loading Optimizations](#model-loading-optimizations)
    - [`mlock` (boolean, optional)](#mlock-boolean-optional)
  - [Display Configuration](#display-configuration)
    - [`truncate_thinking` (integer, optional)](#truncate_thinking-integer-optional)
    - [`truncate_response` (integer, optional)](#truncate_response-integer-optional)
  - [Directory Configuration](#directory-configuration)
    - [`logs_dir` (string, optional)](#logs_dir-string-optional)
    - [`chats_dir` (string, optional)](#chats_dir-string-optional)
  - [Complete Example](#complete-example)
  - [Parameter Priority](#parameter-priority)
  - [Tips](#tips)
  - [Memory Management](#memory-management)
  - [Markdown Rendering](#markdown-rendering)

## Harmony-Specific Parameters (GPT-OSS Models Only)

### `system_model_identity` (string, optional)

The model's system identity/role description. This is used as the Harmony `SystemContent` for GPT-OSS models.

**Example:**

```json
"system_model_identity": "You are {assistant}, an AI researcher and partner. You are intelligent, sensual, and open-minded."
```

**Placeholders:** Supports `<|DATE|>`, `<|DATETIME|>`, and user-defined `{key}` placeholders.

### `reasoning_effort` (string, optional)

The reasoning effort level for the model. Used in Harmony `SystemContent`.

**Values:** `"Low"`, `"Medium"`, `"High"` (or any string)

**Example:**

```json
"reasoning_effort": "Medium"
```

### `conversation_start_date` (string, optional)

The start date for the conversation. Automatically set to current date if not specified.

**Example:**

```json
"conversation_start_date": "<|DATE|>"
```

Or explicit date:

```json
"conversation_start_date": "2025-01-07"
```

**Placeholders:** Supports `<|DATE|>` (replaced with current date in YYYY-MM-DD format).

### `knowledge_cutoff` (string, optional)

The model's knowledge cutoff date. Used in Harmony `SystemContent`.

**Example:**

```json
"knowledge_cutoff": "2024-06"
```

### `developer_instructions` (string, optional)

Developer-level instructions for the model. Used as Harmony `DeveloperContent` for GPT-OSS models. These are higher priority than user messages.

**Example:**

```json
"developer_instructions": "Always respond as {assistant}. Be warm, friendly, and engaging. Use descriptive language."
```

**Placeholders:** Supports `<|DATE|>`, `<|DATETIME|>`, and user-defined `{key}` placeholders.

### `assistant_greeting` (string, optional)

An optional greeting message displayed at the start of a new conversation (before the first user message).

**Example:**

```json
"assistant_greeting": "Hi {user}, it's {assistant}! I'm happy to see you again."
```

**Placeholders:** Supports `<|DATE|>`, `<|DATETIME|>`, and user-defined `{key}` placeholders.

### `example_dialogues` (array of arrays, optional)

Few-shot example conversations to include in the prompt. Each example is a list of message turns (user/assistant pairs). These demonstrate the desired conversation style and are part of the prompt but not sent every time like system/developer instructions.

**Example:**

```json
"example_dialogues": [
  [
    {"role": "user", "content": "Hello, how can you help me?"},
    {"role": "assistant", "content": "Hello {user}! I'm {assistant} and I'm here to help..."}
  ],
  [
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "I don't have access to real-time weather data..."}
  ]
]
```

**Note:** Example dialogues are particularly useful for role-playing assistants, customer service bots, or any scenario where you want to demonstrate the desired conversation style.

## Placeholders

### `placeholders` (object, optional)

User-defined placeholder mappings for text substitution. These are applied to all Harmony fields and message content.

**Example:**

```json
"placeholders": {
  "assistant": "Mia",
  "user": "Michael"
}
```

### Built-in Placeholders

Built-in placeholders are expanded when the prompt is rendered (at generation time), so they always reflect the current date/time.

#### Date Placeholders

- **`<|DATE|>`** - Current date in YYYY-MM-DD format (local time)
  - Example: `2025-01-07`
  
- **`<|DATETIME|>`** - Current date and time in ISO format (local time, seconds precision)
  - Example: `2025-01-07T14:30:45`

#### Time Placeholders

- **`<|TIME|>`** - Current time in 24-hour format (HH:MM:SS, local time)
  - Example: `14:30:45`
  - Note: This is the default time format. Use `<|TIMEZ|>` explicitly for clarity.

- **`<|TIMEZ|>`** - Current time in 24-hour format (HH:MM:SS, local time)
  - Example: `14:30:45`
  - Same as `<|TIME|>`, but explicitly indicates 24-hour format.
  
- **`<|TIMEA|>`** - Current time in 12-hour format with AM/PM (HH:MM:SS AM/PM, local time)
  - Example: `02:30:45 PM`
  - Uses 12-hour format with AM/PM suffix.
  
- **`<|TIMEU|>`** - Current time in 24-hour UTC format (HH:MM:SS UTC)
  - Example: `19:30:45 UTC`
  - Always uses UTC timezone, regardless of local timezone.

### User-defined Placeholders

User-defined placeholders can use **either format** and are replaced with values from the `placeholders` object:

- **`{key}`** - Curly braces format (case-sensitive)
  - Example: If `placeholders` contains `{"assistant": "Mia"}`, then `{assistant}` is replaced with `Mia`
  - Must match the key exactly (case-sensitive)
  
- **`<|KEY|>`** - Angle bracket format (case-insensitive, normalized to uppercase)
  - Example: If `placeholders` contains `{"assistant": "Mia"}`, then `<|ASSISTANT|>`, `<|assistant|>`, or `<|Assistant|>` are all replaced with `Mia`
  - **Note:** Keys are normalized to uppercase for matching, so any case works in the template
  - Consistent with built-in placeholders which are all uppercase (`<|DATE|>`, `<|TIME|>`, etc.)

Both formats are simple string substitutions (no date/time formatting). The `<|KEY|>` format is more forgiving (case-insensitive), while `{key}` format requires exact case matching.

**Example with both formats:**

```json
{
  "placeholders": {
    "assistant": "Mia",
    "ASSISTANT": "Mia",
    "user": "Michael"
  },
  "system_model_identity": "You are {assistant} (or <|ASSISTANT|>). The user is {user}."
}
```

This renders as:

```text
You are Mia (or Mia). The user is Michael.
```

**Important:**

- Built-in placeholders (`<|DATE|>`, `<|TIME|>`, etc.) are **always** checked first and cannot be overridden
- User-defined placeholders in `<|KEY|>` format are **case-insensitive** (normalized to uppercase for matching)
- User-defined placeholders in `{key}` format are **case-sensitive** (must match the key exactly)
- Tool calls (e.g., `browser`, `python`) are not affected by placeholder normalization - they use Harmony message `recipient` fields

**Combined Example:**

```json
{
  "system_model_identity": "You are {assistant}. Today is <|DATE|> at <|TIMEZ|> (local time) or <|TIMEU|> (UTC).",
  "placeholders": {
    "assistant": "Mia",
    "user": "Michael"
  }
}
```

This would render as:

```text
You are Mia. Today is 2025-01-07 at 14:30:45 (local time) or 19:30:45 UTC (UTC).
```

### Placeholder Guide Summary

| Placeholder       | Example output           | Notes                                                    | Example usage                       |
|-------------------|--------------------------|----------------------------------------------------------|-------------------------------------|
| `<\|DATE\|>`      | `2025-01-07`             | Current date (local time), format: `YYYY-MM-DD`          | "Today is <\|DATE\|>"               |
| `<\|DATETIME\|>`  | `2025-01-07T14:30:45`    | ISO 8601 (seconds, local time)                           | "Now: <\|DATETIME\|>"               |
| `<\|TIME\|>`      | `14:30:45`               | 24-hour local time, format: `HH:MM:SS`                   | "Local time: <\|TIME\|>"            |
| `<\|TIMEZ\|>`     | `14:30:45`               | 24-hour local time, same as `<\|TIME\|>`                 | "Local time: <\|TIMEZ\|>"           |
| `<\|TIMEA\|>`     | `02:30:45 PM`            | 12-hour local time with AM/PM, format: `HH:MM:SS AM/PM`  | "12-hour time: <\|TIMEA\|>"         |
| `<\|TIMEU\|>`     | `19:30:45 UTC`           | 24-hour UTC time, format: `HH:MM:SS UTC`                 | "UTC time: <\|TIMEU\|>"             |
| `\{key\}`         | Value from `placeholders`| User-defined, case-sensitive (`\{key\}` matches exactly) | "Agent: \{assistant\}"              |
| `<\|KEY\|>`       | Value from `placeholders`| User-defined, case-insensitive (normalized to uppercase) | "Agent: <\|ASSISTANT\|>"            |

**Notes:**

- All placeholders are expanded at generation time (when the prompt is sent to the model)
- Date/time placeholders reflect the actual current date/time, not when the config was created
- User-defined placeholders (`{key}`) are simple string substitutions
- Placeholders can be used in any Harmony field: `system_model_identity`, `developer_instructions`, `assistant_greeting`, `example_dialogues`, etc.

## Sampling Parameters

### `max_tokens` (integer, optional)

Maximum number of tokens to generate. **Default:** 1024 for Harmony models (to allow for both analysis and final channels), 512 for other models.

**Example:**

```json
"max_tokens": 1024
```

**Note:** For Harmony models, you may need higher values (1024-2048) to allow for both analysis (thinking) and final channel responses.

### `temperature` (float, optional)

Sampling temperature. Controls randomness in generation.

**Range:** 0.0 (deterministic) to 2.0 (very creative)

**Typical Values:**

- `0.0-0.3`: Very focused, deterministic
- `0.7-0.9`: Balanced creativity
- `1.0-1.5`: More creative, diverse
- `>1.5`: Highly creative, may be less coherent

**Example:**

```json
"temperature": 0.8
```

### `top_p` (float, optional)

Nucleus sampling. Keeps the smallest set of tokens whose cumulative probability exceeds `top_p`.

**Range:** 0.0 to 1.0

**Typical Values:**

- `0.9-0.95`: Balanced (recommended)
- `0.5-0.9`: More focused
- `>0.95`: More diverse

**Example:**

```json
"top_p": 0.9
```

### `min_p` (float, optional)

Minimum probability threshold. Filters out tokens with probability less than `min_p` times the probability of the most likely token.

**Range:** 0.0 to 1.0

**Typical Values:**

- `0.0`: No filtering (default)
- `0.05-0.1`: Light filtering
- `>0.1`: More aggressive filtering

**Example:**

```json
"min_p": 0.05
```

### `top_k` (integer, optional)

Top-k sampling. Keeps only the top k most likely tokens at each step.

**Range:** 0 (disabled) to vocabulary size

**Typical Values:**

- `0`: Disabled (use top_p instead)
- `40-100`: Balanced
- `>100`: More diverse

**Example:**

```json
"top_k": 40
```

### `min_tokens_to_keep` (integer, optional)

Minimum number of tokens to keep after filtering (e.g., with min_p or top_k).

**Default:** `1`

**Example:**

```json
"min_tokens_to_keep": 1
```

### `repetition_penalty` (float, optional)

Penalty for repeating tokens. Values > 1.0 reduce repetition, values < 1.0 encourage repetition.

**Range:** > 0.0

**Typical Values:**

- `1.0`: No penalty (default)
- `1.1-1.2`: Light penalty (reduces repetition)
- `>1.2`: Strong penalty (may reduce natural variation)

**Example:**

```json
"repetition_penalty": 1.1
```

### `repetition_context_size` (integer, optional)

Number of previous tokens to consider when calculating repetition penalty.

**Default:** `20`

**Example:**

```json
"repetition_context_size": 20
```

**Note:** Larger values consider more context when penalizing repetition, which may reduce natural repetition of important words/concepts.

### `xtc_probability` (float, optional)

XTC (Experimental Token Control) sampling probability. This controls how often XTC filtering is applied during token generation.

**Range:** 0.0 to 1.0

**Default:** `0.0` (disabled)

**Typical Values:**

- `0.0`: Disabled (default)
- `0.1-0.3`: Light filtering (applied occasionally)
- `0.5-0.7`: Moderate filtering (applied about half the time)
- `>0.7`: Heavy filtering (applied most of the time)

**Example:**

```json
"xtc_probability": 0.5
```

**Note:** This is an experimental feature. XTC sampling filters tokens based on a probability threshold, keeping only tokens with probabilities above the minimum threshold that meets the `xtc_threshold` criterion. Use with caution and test thoroughly with your models.

### `xtc_threshold` (float, optional)

XTC (Experimental Token Control) sampling threshold. This sets the minimum probability threshold that tokens must meet to be considered for sampling when XTC is applied.

**Range:** 0.0 to 0.5

**Default:** `0.0` (disabled)

**Typical Values:**

- `0.0`: Disabled (default)
- `0.05-0.1`: Light filtering (allows more tokens)
- `0.1-0.2`: Moderate filtering (filters low-probability tokens)
- `0.2-0.5`: Heavy filtering (very selective)

**Example:**

```json
"xtc_threshold": 0.1
```

**Note:** Must be used together with `xtc_probability > 0.0` to have any effect. When XTC sampling is triggered (based on `xtc_probability`), only tokens with probabilities above the minimum threshold meeting this `xtc_threshold` will be considered for sampling. This can help reduce low-quality token generation but may also reduce diversity. This is an experimental feature - use with caution.

### `xtc_special_tokens` (array of integers, optional)

List of special token IDs to exclude from XTC filtering. These tokens will always be considered for sampling, even when XTC filtering is applied.

**Default:** `null` (auto-detected when XTC is enabled)

**Auto-detection:**

When `xtc_special_tokens` is `null` or not specified and XTC is enabled (`xtc_probability > 0.0`), special tokens are automatically detected from the tokenizer:

- EOS token ID (end-of-sequence token)
- Newline token ID (`\n`)

This ensures that important control tokens like end-of-sequence markers and newlines can still be generated even when XTC filtering is active.

**Manual override:**

You can manually specify token IDs to exclude from XTC filtering:

```json
"xtc_special_tokens": [2, 13, 220]
```

Where `2` might be the EOS token, `13` might be a newline token, etc.

**Example:**

```json
"xtc_special_tokens": null
```

Or with explicit token IDs (advanced users only):

```json
"xtc_special_tokens": [2, 13]
```

**Note:** Token IDs are model-specific and must match the tokenizer used by your model. If you're unsure of the correct token IDs, leave this as `null` to use auto-detection. This is an experimental feature - use with caution.

## Model Loading Optimizations

### `mlock` (boolean, optional)

Lock model weights in memory using MLX's wired limit (mlock equivalent). Prevents the OS from swapping model weights to disk.

**Default:** `false`

**Example:**

```json
"mlock": true
```

**Requirements:**

- macOS with Metal backend
- Model size must fit within 90% of Metal's recommended working set size

**Note:** This uses MLX's `set_wired_limit()` under the hood, which is the MLX equivalent of mlock.

## Display Configuration

### `truncate_thinking` (integer, optional)

Maximum number of characters to display for thinking/analysis channel content before truncating. Default: `1000`.

**Example:**

```json
"truncate_thinking": 2000
```

**Note:** This only affects display in the chat interface. Full analysis content is still saved in the conversation log.

### `truncate_response` (integer, optional)

Maximum number of characters to display for final response content before truncating. Default: `1000`.

**Example:**

```json
"truncate_response": 2000
```

**Note:** This only affects display in the chat interface. Full response content is still saved in the conversation log.

## Directory Configuration

### `logs_dir` (string, optional)

Directory path for debug logs. Default: `"logs"`.

**Example:**

```json
"logs_dir": "logs"
```

Or use a custom path:

```json
"logs_dir": "/var/log/mlx-harmony"
```

**Note:** The directory is created automatically if it doesn't exist. Debug logs (from `--debug` or `--debug-file`) are written to `logs_dir/prompt-debug.log` by default.

### `chats_dir` (string, optional)

Directory path for chat history files. Default: `"logs"`.

**Example:**

```json
"chats_dir": "logs"
```

Or use a custom path:

```json
"chats_dir": "conversations"
```

**Note:** The directory is created automatically if it doesn't exist. Chat files are saved as `chats_dir/<chat_name>.json` when using `--chat <name>`.

## Complete Example

```json
{
  "system_model_identity": "You are {assistant}, an AI assistant.",
  "reasoning_effort": "Medium",
  "conversation_start_date": "<|DATE|>",
  "knowledge_cutoff": "2024-06",
  "developer_instructions": "Be helpful, friendly, and concise.",
  "assistant_greeting": "Hello {user}, I'm {assistant}. How can I help you today?",
  "example_dialogues": [
    [
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hello {user}! How can I help?"}
    ]
  ],
  "placeholders": {
    "assistant": "Dave",
    "user": "Morgan"
  },
  "max_tokens": 1024,
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 40,
  "min_p": 0.0,
  "min_tokens_to_keep": 1,
  "xtc_probability": 0.0,
  "xtc_threshold": 0.0,
  "xtc_special_tokens": null,
  "repetition_penalty": 1.0,
  "repetition_context_size": 20,
  "mlock": false,
  "truncate_thinking": 1000,
  "truncate_response": 1000,
  "logs_dir": "logs",
  "chats_dir": "logs"
}
```

## Parameter Priority

When a parameter is specified in multiple places, the priority is:

1. **CLI arguments** (highest priority) - e.g., `--max-tokens 2048`
2. **Loaded conversation metadata** - From saved conversation JSON
3. **Prompt config JSON** - From `--prompt-config` file
4. **Default values** (lowest priority) - Built-in defaults

## Tips

- **For Harmony models**: Set `max_tokens` to 1024-2048 to allow for both analysis and final channel responses
- **For creative tasks**: Use higher `temperature` (0.9-1.2) with `top_p` 0.9-0.95
- **For focused tasks**: Use lower `temperature` (0.3-0.7) with `top_p` 0.7-0.9
- **To reduce repetition**: Set `repetition_penalty` to 1.1-1.2
- **For stable memory**: Enable `mlock: true` on macOS (prevents swapping)
- **For long responses**: Increase `truncate_thinking` and `truncate_response` if you want to see more content in the chat interface
- **For organization**: Use separate `logs_dir` and `chats_dir` to organize output files

## Memory Management

For detailed information about memory management, including wired memory (mlock) and considerations for loading multiple models, see [Memory Management Guide](./MEMORY_MANAGEMENT.md).

## Markdown Rendering

The chat CLI (`mlx-harmony-chat`) automatically renders assistant responses as markdown using the `rich` library, similar to `glow` or `mdless`. This provides beautiful formatting for:

- Headers (`#`, `##`, `###`, etc.)
- Lists (numbered and bulleted)
- Code blocks (with syntax highlighting)
- Bold and italic text
- Blockquotes

**Example:** If the model generates:

```text
### Simplifying Optimal Loading on macOS
1. **Checkpointing**: Instead of loading...
```

It will be automatically formatted with proper styling, newlines, and colors.

**Disable markdown rendering:** Use the `--no-markdown` flag:

```bash
mlx-harmony-chat --model openai/gpt-oss-20b --no-markdown
```

This is useful if you prefer plain text output or are piping output to other tools.

---

[← Back to README](../README.md)
