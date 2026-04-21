import json
import os
from argparse import ArgumentParser
from src.loader import load_json_file, load_vocabulary
from src.models import FunctionDefinition, Prompt, FunctionCallResult
from src.constrained import (
    build_prompt_for_function,
    build_prompt_for_argument,
    generate_function_name,
    generate_argument_value
)
from llm_sdk import Small_LLM_Model


def main() -> None:
    """Entry point for the function calling pipeline."""
    parser = ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json"
    )
    args = parser.parse_args()

    # load files
    functions = [
        FunctionDefinition(**fn)
        for fn in load_json_file(args.functions_definition)
    ]
    prompts = [
        Prompt(**p)
        for p in load_json_file(args.input)
    ]

    # load model and vocabulary once
    model = Small_LLM_Model()
    id_to_str, _ = load_vocabulary(model.get_path_to_vocab_file())

    results = []

    for p_obj in prompts:
        print(f"\nProcessing: {p_obj.prompt}")
        try:
            # step 1 — generate function name
            fn_prompt = build_prompt_for_function(
                p_obj.prompt, functions
            )
            fn_prompt_ids = model.encode(fn_prompt)[0].tolist()

            chosen_name = generate_function_name(
                model, fn_prompt_ids, id_to_str, functions
            )
            print(f"  Function: {chosen_name}")

            # find the chosen function definition
            chosen_fn = next(
                fn for fn in functions if fn.name == chosen_name
            )

            # step 2 — generate each argument
            parameters = {}
            for arg_name, arg_def in chosen_fn.parameters.items():
                arg_prompt = build_prompt_for_argument(
                    p_obj.prompt,
                    chosen_name,
                    arg_name,
                    arg_def.type
                )
                arg_prompt_ids = model.encode(arg_prompt)[0].tolist()

                value = generate_argument_value(
                    model,
                    arg_prompt_ids,
                    id_to_str,
                    arg_def.type
                )
                parameters[arg_name] = value
                print(f"  {arg_name} = {repr(value)}")

            # step 3 — build result
            result = FunctionCallResult(
                prompt=p_obj.prompt,
                name=chosen_name,
                parameters=parameters
            )
            results.append(result.model_dump())

        except Exception as e:
            print(f"  Error: {e}")

    # write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
