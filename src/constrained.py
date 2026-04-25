from typing import List
from .models import FunctionDefinition


def build_prompt_for_function(
    user_prompt: str,
    functions: List[FunctionDefinition]
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
    lines.append('\nCall the correct function.')
    lines.append('\nFunction name: "')
    return "\n".join(lines)


def build_prompt_for_argument(
    user_prompt: str,
    function_name: str,
    arg_name: str,
    arg_type: str
) -> str:
    """Build the prompt for one argument value.

    Args:
        user_prompt: the natural language request from the user.
        function_name: the chosen function name.
        arg_name: the name of the argument to fill.
        arg_type: the type of the argument (string, number, boolean).

    Returns:
        A prompt string ending so the LLM starts generating
        the argument value immediately.
    """
    lines = [f"User request: {user_prompt}"]
    lines.append(f"Function: {function_name}")
    if arg_type == "string":
        lines.append(f'Fill argument "{arg_name}" (type: {arg_type}): "')
    else:
        lines.append(f'Fill argument "{arg_name}" (type: {arg_type}): ')
    return "\n".join(lines)
