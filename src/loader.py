import sys
import json
from argparse import ArgumentParser
from typing import Tuple, List
from src.models import FunctionDefinition, Prompt


def load_vocabulary(
    path: str
) -> Tuple[dict[int, str], dict[str, int]]:
    """Load the vocabulary JSON file and return both lookup directions.

    Args:
        path: path to the vocabulary JSON file from llm_sdk.

    Returns:
        id_to_str: maps token ID (int) to its string representation.
        str_to_id: maps token string to its ID (int).
    """
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found: {path}",
              file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid vocabulary JSON: {e}",
              file=sys.stderr)
        sys.exit(1)
    str_to_id: dict[str, int] = {k: int(v) for k, v in raw.items()}
    id_to_str: dict[int, str] = {int(v): k for k, v in raw.items()}
    return id_to_str, str_to_id


def load_json_file(path: str) -> list | dict:
    """Load and parse a JSON file gracefully."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: No permission to access the file: {path}",
              file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def parse_a_json_file() -> Tuple[List[FunctionDefinition], List[Prompt]]:
    """Load and validate both input files."""
    parser = ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--module",
                        default="Qwen/Qwen3-0.6B")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calls.json")
    args = parser.parse_args()
    raw_functions = load_json_file(args.functions_definition)
    raw_prompts = load_json_file(args.input)
    functions = [FunctionDefinition(**fn) for fn in raw_functions]
    prompts = [Prompt(**p) for p in raw_prompts]
    return args, functions, prompts
