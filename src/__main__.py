import json
import os
import sys
from src.loader import parse_a_json_file, load_vocabulary
from src.models import FunctionCallResult, JsonValidationError
from src.prompt_build import (
    build_prompt_for_function,
    build_prompt_for_argument,
)
from src.generate import generate_function_name, generate_argument_value
from llm_sdk import Small_LLM_Model


def main() -> None:
    """Entry point for the function calling pipeline."""
    try:
        args, functions, prompts = parse_a_json_file()
        for fn in functions:
            if fn.name.strip() == "":
                fn.name = "fn_didn't_have_name"
        for pr in prompts:
            if pr.prompt.strip() == "":
                pr.prompt = "empthy"
        model = Small_LLM_Model(args.model)
    except (OSError, JsonValidationError) as e:
        print(f"❌ [ERROR]: {e}")
        sys.exit(1)
    id_to_str = load_vocabulary(model.get_path_to_vocab_file())
    results = []
    i = 0
    for p_obj in prompts:
        try:
            i += 1
            if p_obj.prompt == "empthy":
                raise ValueError("No valid prompt matched this prompt.")
            print(f"\nProcessing[{i}/{len(prompts)}]:")
            print(f"   Prompt: {p_obj.prompt}")
            # generate function name
            fn_prompt = build_prompt_for_function(p_obj.prompt, functions)
            fn_prompt_ids = model.encode(fn_prompt)[0].tolist()
            chosen_name = generate_function_name(
                model, fn_prompt_ids, id_to_str, functions
            )
            if chosen_name == "fn_didn't_have_name":
                raise ValueError("No valid function matched this prompt.")
            print(f"    ✅ Function: {chosen_name}")
            chosen_fn = next(
                fn for fn in functions if fn.name == chosen_name
            )
            # generate each argument
            parameters: dict = {}
            for arg_name, arg_def in chosen_fn.parameters.items():
                arg_prompt = build_prompt_for_argument(
                    p_obj.prompt,
                    chosen_name,
                    arg_name,
                    arg_def.type,
                    already_filled=parameters,
                )
                arg_prompt_ids = model.encode(arg_prompt)[0].tolist()
                value = generate_argument_value(
                    model,
                    arg_prompt_ids,
                    id_to_str,
                    arg_def.type,
                )
                parameters[arg_name] = value
                print(f"        {arg_name} = {repr(value)}")
            result = FunctionCallResult(
                prompt=p_obj.prompt,
                name=chosen_name,
                parameters=parameters,
            )
            results.append(result.model_dump())
        except Exception as e:
            print(f" ❌ Error: {e}")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved to {args.output}")


if __name__ == "__main__":
    main()
