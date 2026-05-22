*This project has been created as part of the 42 curriculum by hlaaz.*

# call me maybe

## Description

**call me maybe** is a function-calling pipeline that translates natural-language
prompts into structured function calls using a small Large Language Model (LLM).

Given a prompt like `"What is the sum of 40 and 2?"`, the system does not answer
`42`. Instead it produces:

```json
{
  "name": "fn_add_numbers",
  "parameters": { "a": 40.0, "b": 2.0 }
}
```

The project explores **constrained decoding** — a technique that guides an LLM's
output token-by-token to guarantee valid, schema-compliant results even with a tiny
0.6 B parameter model (Qwen/Qwen3-0.6B).

---

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone git@github.com:hlaaz/call-me-maybe.git
cd call-me-maybe

# Copy the llm_sdk package into the project root (provided separately)
cp -r /path/to/llm_sdk ./llm_sdk

# Install dependencies
make install
# or manually:
uv sync
```

### Running

```bash
make run
# or manually:
uv run python -m src
```

Custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json \
  --model Qwen/Qwen3-0.6B
```

### Other Makefile targets

| Command        | Description                              |
|----------------|------------------------------------------|
| `make install` | Install dependencies with uv             |
| `make run`     | Run the pipeline                         |
| `make debug`   | Run with Python debugger (pdb)           |
| `make lint`    | Run flake8 + mypy                        |
| `make clean`   | Remove `__pycache__` and `.mypy_cache`   |

---

## Example Usage

**Input** `data/input/function_calling_tests.json`:

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Replace all vowels in 'hello' with asterisks" }
]
```

**Output** `data/output/function_calls.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Replace all vowels in 'hello' with asterisks",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "hello",
      "regex": "[aeiouAEIOU]",
      "replacement": "*"
    }
  }
]
```

**Console output:**

```
✅ Processing[1/2]:
   Prompt: What is the sum of 2 and 3?
    Function: fn_add_numbers
        a = 2.0
        b = 3.0
💾 Saved to data/output/function_calls.json
```

---

## Algorithm Explanation

The pipeline runs in three stages for each prompt:

### Stage 1 — Function Selection

A prompt listing all available functions and their descriptions is built and fed to
the LLM. Constrained decoding restricts token selection so only characters that form
a valid prefix of a known function name can be emitted. Generation stops the moment
a closing `"` is the highest-scoring valid token, guaranteeing the output is always
an exact function name.

```
logits = model.get_logits_from_input_ids(prompt_ids)
for each token_id:
    if token does NOT extend a valid function name prefix:
        logits[token_id] = -inf          # kill invalid token
chosen = argmax(logits)                  # pick best valid token
```

### Stage 2 — Argument Extraction

For each parameter of the chosen function a new prompt is built. The prompt uses a
**JSON-completion format** with concrete few-shot examples placed immediately before
the generation point, maximising their influence on the small model:

```
Input: "Replace all vowels in 'this is a test' with asterisks"
JSON: {"name": "fn_substitute_string_with_regex",
       "parameters": {"source_string": "this is a test",
                      "regex": "[aeiouAEIOU]", "replacement": "*"}}

Input: "<actual user prompt>"
JSON:
{
    "name": "fn_substitute_string_with_regex",
    "parameters": {"source_string": "Programming is fun", "regex": "
                                                                      ^ model writes here
```

Constrained decoding enforces type correctness:

- **string** — any token allowed; stops on closing `"`
- **number / integer** — only digit tokens, optional leading `-`, optional `.`
- **boolean** — only tokens that extend `"true"` or `"false"`

Already-generated arguments are inlined into the partial JSON so the model never
repeats a value assigned to an earlier parameter.

### Stage 3 — Output

Results are collected into a list of `FunctionCallResult` objects and serialised to
a JSON file that is always valid and schema-compliant.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Constrained decoding over prompting alone | A 0.6 B model produces valid JSON only ~30 % of the time without constraints; constrained decoding raises this to ~100 %. |
| JSON-completion prompt format | Small models pattern-match better than they follow instructions; showing a complete JSON example right before the generation point is more effective than writing rules. |
| `already_filled` context in argument prompts | Without seeing previously generated arguments, the model tends to repeat the first value for every subsequent parameter (e.g. `b = a` for addition). |
| `JsonValidationError(Exception)` | Pydantic's `ValidationError` cannot be subclassed with a plain string argument; inheriting from `Exception` gives clean, readable error messages. |

---

## Performance Analysis

Tested on the 11 reference prompts with Qwen/Qwen3-0.6B:

| Metric | Result |
|--------|--------|
| Function selection accuracy | 11 / 11 (100 %) |
| Argument extraction accuracy | ~10 / 11 (≈ 91 %) |
| Output JSON validity | 100 % — constrained decoding guarantees it |
| Runtime (11 prompts, CPU) | ~40 – 70 seconds |

The single failure mode observed is the regex argument for complex word-substitution
prompts, where the model occasionally adds unnecessary grouping. This is a prompting
challenge inherent to 0.6 B models and does not affect JSON validity.

---

## Challenges Faced

**1. Token boundaries vs. expected characters**

The tokenizer does not emit one character per token. A closing `"` may arrive
embedded in a longer token such as `"Answer:`. The string termination check was
updated to scan for `"` anywhere inside the token and split on it:

```python
if '"' in next_str:
    accumulated += next_str.split('"')[0]
    return accumulated
```

**2. Repeated argument values (`b = a`)**

Without context, the model picks the most probable next number — which is the one
it just generated. Passing `already_filled` into the prompt shows the model what has
already been assigned and forces it to look for a different value.

**3. Regex prompt reliability**

A 0.6 B model ignores abstract rules like "no `.*` wrapper". Switching to a
JSON-completion format with concrete examples (input → full JSON) that the model
pattern-matches against solved this more reliably than any instruction-based approach.

---

## Testing Strategy

1. **Reference prompts** — the 11 prompts in `function_calling_tests.json` were run
   after every change to verify no regression.
2. **Edge cases tested manually:**
   - Uppercase replacements (`replace with NUMBERS`)
   - Quoted word substitution (`substitute the word 'cat' with 'dog'`)
   - Vowel and digit regex patterns
3. **JSON validity** — output file was parsed with `json.load` after every run to
   confirm the constrained decoder never produced malformed JSON.
4. **Error handling** — missing files, invalid JSON input, and unknown function names
   were tested to confirm graceful error messages and clean exits.

---

## Resources

### References

- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-0.6B) — the LLM used in this project
- [Outlines library](https://github.com/dottxt-ai/outlines) — inspiration for constrained decoding (not used directly)
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/) — data validation and schema enforcement
- [Byte-Pair Encoding (BPE)](https://huggingface.co/learn/nlp-course/chapter6/5) — tokenization algorithm used by Qwen
- [JSON specification](https://www.json.org/json-en.html)

### AI Usage

Claude (Anthropic) was used throughout this project for:

- **Debugging** — identifying why the tokenizer emitted multi-character tokens
- **Prompt engineering** — iterating on the few-shot JSON-completion format to
  improve regex and replacement extraction reliability
- **Code review** — checking type hints, docstrings, and flake8 compliance
- **README drafting** — structuring this document (reviewed and edited by hlaaz)

All AI-generated code was reviewed, tested, and understood before being included.
PYEOF
