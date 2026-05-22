import sys
import json
from argparse import ArgumentParser, Namespace
from typing import Tuple, List, Any, Dict, Union
from src.models import FunctionDefinition, Prompt, JsonValidationError
from pydantic import ValidationError


def load_vocabulary(path: str) -> Dict[int, str]:
    """Load the vocabulary JSON file and return id -> token-string mapping.

    Args:
        path: path to the vocabulary JSON file from llm_sdk.

    Returns:
        id_to_str: maps token ID (int) to its string representation.
    """
    try:
        with open(path, "r") as f:
            raw: Dict[str, int] = json.load(f)
    except FileNotFoundError:
        print(f"❌[Error]: Vocabulary file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌[Error]: Invalid vocabulary JSON: {e}")
        sys.exit(1)
    id_to_str: Dict[int, str] = {v: k for k, v in raw.items()}
    return id_to_str


def load_json_file(path: str) -> Union[List[Any], Dict[str, Any]]:
    """Load and parse a JSON file gracefully.

    Args:
        path: path to the JSON file.

    Returns:
        The parsed JSON value (list or dict).
    """
    try:
        with open(path, "r") as f:
            data: Union[List[Any], Dict[str, Any]] = json.load(f)
            return data
    except FileNotFoundError:
        print(f"❌[Error]: File not found: {path}")
        sys.exit(1)
    except PermissionError:
        print(f"❌[Error]: No permission to access the file: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌[Error]: Invalid JSON in {path}: {e}")
        sys.exit(1)


def parse_a_json_file() -> Tuple[
    Namespace,
    List[FunctionDefinition], List[Prompt]
        ]:
    """Load and validate both input files.

    Returns:
        A 3-tuple of (args Namespace, list of FunctionDefinition,
        list of Prompt).
    """
    parser = ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json")
    parser.add_argument("--model",
                        default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    raw_functions = load_json_file(args.functions_definition)
    raw_prompts = load_json_file(args.input)
    try:
        functions = [FunctionDefinition(**fn)
                     for fn in raw_functions if isinstance(fn, dict)]
        prompts = [Prompt(**p) for p in raw_prompts if isinstance(p, dict)]
    except ValidationError as e:
        raise JsonValidationError(f"Invalid JSON schema:\n{e}")
    return args, functions, prompts
