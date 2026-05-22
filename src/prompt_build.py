from typing import List, Dict, Any, Optional
from src.models import FunctionDefinition


def _format_value(value: object) -> str:
    """Format a Python value as its JSON representation.

    Args:
        value: the Python value (str, bool, or numeric).

    Returns:
        A JSON-compatible string representation.
    """
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _regex_hint(user_prompt: str) -> str:
    """Return a targeted hint for the regex argument.

    Args:
        user_prompt: the original natural-language request.

    Returns:
        A short hint sentence tailored to the user prompt.
    """
    low = user_prompt.lower()
    if "vowels" in low:
        return 'use the regex pattern [aeiouAEIOU] for vowels'
    if "number" in low or "digit" in low:
        return "use the regex pattern [0-9]+ for numbers or digits"
    if "word" in low or "substitute" in low or "replace" in low:
        return (
            "extract only the exact literal word to match, "
            "no capturing groups, no lookaheads, no alternation"
        )
    return "extract only the exact word or pattern to match"


def build_prompt_for_function(
    user_prompt: str,
    functions: List[FunctionDefinition],
) -> str:
    """Build the prompt for function name selection.

    Args:
        user_prompt: the natural language request from the user.
        functions: list of available function definitions.

    Returns:
        A prompt string ending with an opening quote so the LLM
        starts generating the function name immediately.
    """
    lines = ["You are a function dispatcher.\n"]
    lines.append("Available functions:")
    for fn in functions:
        lines.append(f"- {fn.name}: {fn.description}")
    lines.append(f"\nUser request: {user_prompt}")
    lines.append("\nCall the correct function.")
    lines.append('\nFunction name: "')
    return "\n".join(lines)


def build_prompt_for_argument(
    user_prompt: str,
    function_name: str,
    arg_name: str,
    arg_type: str,
    already_filled: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the prompt for one argument value.

    Uses a JSON-completion format with concrete few-shot examples so
    the model can pattern-match rather than reason from rules.

    Args:
        user_prompt: the natural language request from the user.
        function_name: the chosen function name.
        arg_name: the name of the argument to fill.
        arg_type: the type of the argument (string, number, boolean).
        already_filled: arguments already generated in this call,
                        prevents the model from repeating earlier values.

    Returns:
        A prompt string ending so the LLM starts generating
        the argument value immediately.
    """
    # Build the already-filled part of the partial JSON
    already_json = ""
    if already_filled:
        pairs = ", ".join(
            f'"{k}": {_format_value(v)}'
            for k, v in already_filled.items()
        )
        already_json = f"{pairs}, "
    value_start = '"' if arg_type == "string" else ""
    hint = _regex_hint(user_prompt) if arg_name == "regex" else ""
    return (
        f"Task: Complete the JSON function call.\n"
        f"Function: {function_name}\n"
        f"\nRULES:\n"
        f"1. Write the SHORTEST possible value.\n"
        f"2. For numbers output digits only and stop immediately.\n"
        f'3. For vowels output EXACTLY "[aeiouAEIOU]" and stop.\n'
        f"4. For digits/numbers output EXACTLY [0-9]+ and stop.\n"
        f'5. ALWAYS close strings with a double quote (").\n'
        f"6. NO .* wrapper. NO capturing groups ().\n"
        f"\n--- EXAMPLES ---\n"
        f"Input: \"Replace all numbers in 'Phone 555' with NUMBERS\"\n"
        f'JSON: {{"name": "fn_substitute_string_with_regex", '
        f'"parameters": {{"source_string": "Phone 555", '
        f'"regex": "([0-9]+)", "replacement": "NUMBERS"}}}}\n'
        f"\n"
        f"Input: \"Substitute the word 'cat' with 'dog' in 'the cat sat'\"\n"
        f'JSON: {{"name": "fn_substitute_string_with_regex", '
        f'"parameters": {{"source_string": "the cat sat", '
        f'"regex": "cat", "replacement": "dog"}}}}\n'
        f"\n"
        f"Input: \"Replace all vowels in 'this is a test' with asterisks\"\n"
        f'JSON: {{"name": "fn_substitute_string_with_regex", '
        f'"parameters": {{"source_string": "this is a test", '
        f"'regex': '[aeiouAEIOU]', 'replacement': '*'}}}}\n"
        f"\n"
        f"\n--- NOW COMPLETE ---\n"
        + (f"Hint: {hint}\n" if hint else "")
        + f'Input: "{user_prompt}"\n'
        f"JSON:\n"
        f"{{\n"
        f'    "name": "{function_name}",\n'
        f'    "parameters": {{{already_json}"{arg_name}": {value_start}'
    )
