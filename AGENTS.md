# Operational Guardrails (MANDATORY)

- Work ONLY inside `src/connectomeai/`, `tests/`, and `docs/`.
- YOU may update the Markdown files in the project root, such as README.md, CHANGELOG.md, CONTRIBUTING.md, etc.
- NEVER touch:
  - `.git/`
  - `.cursor/` (or recreate it if removed)
  - `tmp/` (may recreate if needed, but don’t delete)
  - `config/`, `.connectomeai/` runtime dirs (except adding missing configs if needed).
- NEVER:
  - Delete or modify `.gitignore`, `pyproject.toml`, `pytest.ini` without explicit instruction.
  - Install new frameworks (no Trio, no new bus, no random deps).
  - The `models` directory contains LLM model weights and is very large. Do not grep or search that directory.
  - The `external` directory contains external repositories, not directly related to this repository, though has components we need for building the project, do not grep or search that directory.
- ALWAYS:
  - Make small, incremental changes.
  - After a logical batch, STOP and run:
    - `git diff --stat`
    - `python -m pytest -q --maxfail=3 --disable-warnings || true`
  - If tests or imports get *worse*, REVERT that batch instead of patching around it.

# Project Details for the ConnectomeAI repository

1. Scope
   - Only modify files under `src/connectomeai/` **unless explicitly told otherwise**.
   - Do NOT create or modify any directories at the repo root except when explicitly instructed.
   - Do NOT touch `.git`, `.gitignore`, `.cursor`, `agents.md`, or ops/tooling configs unless explicitly instructed.
   - Do not delete or copy over files in `tmp`, you may edit files there if needed.

2. Architectural boundaries
   - Core framework:
     - `connectomeai/core`, `connectomeai/ipc`, `connectomeai/admin`, `connectomeai/api`, `connectomeai/orch`, `connectomeai/sdk`, `connectomeai/schemas`
       are the platform and management layers.
     - Do NOT introduce extension-specific logic into these packages.
     - Do NOT add new top-level packages under `src/connectomeai` beyond:
       - `actors`, `admin`, `api`, `cli`, `core`, `ipc`, `main.py`, `orch`, `schemas`, `sdk`, `templates`, `ui`, `ui2`, `extensions` (if present and explicitly requested).
   - Extensions:
     - Provider-specific or feature-specific implementations (e.g. ElevenLabs TTS) must live in a self-contained package:
       - For now, keep ElevenLabs at `connectomeai/actors/tts/elevenlabs/` as in the current tree.
       - When reorganizing, use `connectomeai/extensions/<provider>/...` but ONLY when explicitly instructed.
     - An extension must:
       - Contain its own client code for external APIs/endpoints.
       - Contain its own agent/worker process logic.
       - Contain any default config, schemas, and UI fragments that belong only to that extension.
     - The core MUST interact with extensions only through stable, minimal abstractions (e.g. registry, config, IPC), not deep relative imports.

3. Imports and error handling
   - Do NOT use `try: import ... except ImportError:` to hide missing core dependencies.
     - If something is required to run, import it normally and fail fast.
     - Only use optional imports behind a clearly documented feature flag or extras dependency.
   - No new relative-import spaghetti. Prefer absolute imports within `connectomeai.*`.
   - When moving code:
     - Update all affected imports.
     - Run `python -m compileall src` and ensure there are no ImportErrors.
   - Avoid silent fallbacks or None returns on error paths.
     - If a required value cannot be resolved, log a clear error and raise a runtime exception so failures are visible and stop execution.
     - The caller should handle the exception and either retry or fail gracefully, and in some cases exit the application.

4. UI and UI2
   - `connectomeai/ui` is the legacy UI. Do NOT delete it yet.
   - `connectomeai/ui2` is the JSON-driven UI layer with pluggable backends.
   - Allowed work on UI2:
     - Implement or fix `ui2/types.py`, `ui2/loader.py`, `ui2/registry.py`, `ui2/service.py`, and `ui2/backends/*` so they:
       - Validate JSON specs via `UISpec`.
       - Load UI specs from `src/connectomeai/ui2/assets/*.json`.
       - Select backends in a clean, dependency-optional way.
     - Add JSON specs for existing admin pages under `ui2/assets/` ONLY if they validate against `UISpec`.
   - Do NOT:
     - Introduce new UI frameworks or dependencies beyond what’s explicitly requested.
     - Copy-paste large amounts of unused demo code.

5. IPC / Bus
   - Use NATS-based bus implementation as the primary path (where present).
   - Do NOT reintroduce Trio or complex in-memory bus implementations unless explicitly requested as a pure test helper.
   - Test helpers must live under `tests/` unless explicitly required at runtime.

6. Behavior changes
   - Prefer deletion and simplification over adding new layers.
   - Do NOT introduce new services, protocols, agents, or providers unless:
     - They are requested explicitly, AND
     - They are fully wired, tested, and documented.
   - Run tests relevant to the changed area. At minimum:
     - `python -m compileall src`
     - `CONNECTOMEAI_BASE_DIR="$PWD/.connectomeai" python -m pytest -q --maxfail=3 --disable-warnings || true`

7. Communication
   - When you propose changes, show:
     - Which files are added/removed/modified.
     - Why each change is necessary in terms of the architecture above.
   - Do NOT silently add new dependencies, directories, or background services.

## Coding and project standards

- Use f-strings for string interpolation with lazy formatting
- Use type hints for all functions and variables
- Use Pydantic for data validation, configuration, serialization and persistence
- Use project logging module for logging
- Use project config module for configuration
- Import modules in proper scope, not inline in functions
- Imports should not be done in functions or other code unless there is a very good reason to do so
- Do not use relative imports, this is a package to be installed using `pip -e .`
- When opening files or writing files, specify encoding UTF-8
- Place any tests in the project test directory
- Make sure all `__init__.py` files are correct
- Make sure the project is correct. Including package and install requirements and test coverage
  - TOML file, and other python setup files.
- Use the command line tool `connectomeai` for starting, stopping and restarting the system.
- Modules/Python files should be no more than 300-4400 lines long, if they are too long they should be broken into smaller specialized files
- Do not over-engineer things, like excessive try/except blocks, and excessive nesting of if/else blocks.
- Modules should do one things and one things well
- This project is broken into two pieces - infrastructure and Agents/Actors/Extensions
- Neither should be tightly coupled to the other
- Adding new extensions/actors/agents should not require modification to the infrastructure
- Adding new infrastructure should not require modification to the agents/actors/extensions
- Use kwargs instead of positional arguments to avoid errors
- Use type hints for all functions and variables

## Markdown Document Standards

- Every Markdown doc starts with `# Title`, then `**Created**` and `**Updated**` dates (update the latter whenever the doc changes).
- Surround headings, lists, and fenced code blocks with blank lines; specify a language on fences (` ```bash `, ` ```text `, etc.).
- Use Markdown checkboxes (`- [ ]`, `- [x]`) instead of emoji for task/status lists.
- Whenever you mention another file or doc, use a relative Markdown link so it's clickable - [Document or File Name](ralative/direct link to document or file)
- Prefer small, single-purpose docs (<= ~500 lines). If a doc grows beyond that, split by topic or scope and link between them. For example:
  - System Overview (Refers to sub-guides)
    - User Guide
    - Developer Guide
    - Technical Reference
    - Best Practices
    - Troubleshooting
    - FAQ
- At "final draft" (or before committing), run `markdownlint` on the file and fix reported issues.
