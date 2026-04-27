import numpy as np
from typing import List
from .models import FunctionDefinition


def is_valid_function_token(
    token_str: str,
    accumulated: str,
    functions: List[FunctionDefinition]
) -> bool:
    """Check if a token is valid during function name generation.

    Args:
        token_str: the string this token produces.
        accumulated: what has been generated so far.
        functions: list of available function definitions.

    Returns:
        True if the token can legally be emitted now.
    """
    valid_names = [fn.name for fn in functions]
    clean = token_str.replace("Ġ", " ").replace("Ċ", "")

    if clean == '"':
        return accumulated in valid_names

    candidate = accumulated + clean
    return any(name.startswith(candidate) for name in valid_names)


def is_valid_argument_token(
    token_str: str,
    accumulated: str,
    arg_type: str,
    has_decimal: bool
) -> bool:
    """Check if a token is valid during argument value generation.

    Args:
        token_str: the string this token produces.
        accumulated: what has been generated so far.
        arg_type: the type of the argument (string, number, boolean).
        has_decimal: whether a decimal point has already appeared.

    Returns:
        True if the token can legally be emitted now.
    """
    clean = token_str.replace("Ġ", " ").replace("Ċ", "")

    if arg_type == "string":
        return True

    if arg_type == "number":
        if clean == "-":
            return accumulated == ""
        if clean == ".":
            return not has_decimal
        if clean.isdigit():
            return True
        return False

    if arg_type == "boolean":
        candidate = accumulated + clean
        return (
            "true".startswith(candidate)
            or "false".startswith(candidate)
        )

    return False


def generate_function_name(
    model: object,
    prompt_ids: List[int],
    id_to_str: dict,
    functions: List[FunctionDefinition],
    # max_tokens: int = 50
) -> str:
    """Generate a function name using constrained decoding.

    Args:
        model: the LLM model instance.
        prompt_ids: encoded prompt token IDs as a plain list.
        id_to_str: vocabulary mapping ID to string.
        functions: list of available function definitions.
        max_tokens: maximum tokens to generate.

    Returns:
        The chosen function name as a string.
    """
    current_ids = list(prompt_ids)
    accumulated = ""

    # for _ in range(max_tokens):
    while True:
        # 1 — get logits from model
        logits = model.get_logits_from_input_ids(current_ids)
        logits_np = np.array(logits)

        # 2 — kill invalid tokens
        for token_id, token_str in id_to_str.items():
            if not is_valid_function_token(
                token_str, accumulated, functions
            ):
                logits_np[int(token_id)] = float("-inf")

        # 3 — pick best valid token
        next_id = int(np.argmax(logits_np))
        next_str = id_to_str[next_id].replace("Ġ", " ").replace("Ċ", "")

        # 4 — closing quote means we are done
        if next_str == '"':
            return accumulated

        # 5 — accumulate and continue
        accumulated += next_str
        current_ids.append(next_id)

    raise ValueError(
        f"Could not generate function name in {max_tokens} tokens"
    )


def generate_argument_value(
    model: object,
    prompt_ids: List[int],
    id_to_str: dict,
    arg_type: str,
    # max_tokens: int = 100
) -> str | float | bool:
    """Generate one argument value using constrained decoding.

    Args:
        model: the LLM model instance.
        prompt_ids: encoded prompt token IDs as a plain list.
        id_to_str: vocabulary mapping ID to string.
        arg_type: the type of the argument (string, number, boolean).
        max_tokens: maximum tokens to generate.

    Returns:
        The argument value in the correct Python type.
    """
    current_ids = list(prompt_ids)
    accumulated = ""
    has_decimal = False

    # for _ in range(max_tokens):
    while True:
        # 1 — get logits
        logits = model.get_logits_from_input_ids(current_ids)
        logits_np = np.array(logits)

        # 2 — kill invalid tokens
        for token_id, token_str in id_to_str.items():
            if not is_valid_argument_token(
                token_str, accumulated, arg_type, has_decimal
            ):
                logits_np[int(token_id)] = float("-inf")

        # 3 — pick best valid token
        next_id = int(np.argmax(logits_np))
        next_str = id_to_str[next_id].replace("Ġ", " ").replace("Ċ", "")

        # 4 — check if done

        if arg_type == "string":
            if next_str == '"':
                return accumulated
            accumulated += next_str
            current_ids.append(next_id)
            continue

        if arg_type == "boolean":
            accumulated += next_str
            if accumulated in ("true", "false"):
                return accumulated == "true"
            current_ids.append(next_id)
            continue

        if arg_type == "number":
            if next_str == ".":
                has_decimal = True
            accumulated += next_str
            current_ids.append(next_id)

            # peek at next token to see if number is done
            logits2 = model.get_logits_from_input_ids(current_ids)
            logits_np2 = np.array(logits2)
            next_id2 = int(np.argmax(logits_np2))
            next_str2 = id_to_str[next_id2].replace("Ġ", " ")
            if not is_valid_argument_token(
                next_str2, accumulated, "number", has_decimal
            ):
                return float(accumulated)
            continue

    raise ValueError(
        f"Could not generate argument in {max_tokens} tokens"
    )
