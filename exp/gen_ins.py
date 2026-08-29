import os
import sys
import argparse
import json
import time
import math
import random
import requests
from time import sleep
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import str_utils

# Resolve the config path relative to this file so the script works from any CWD.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'model_configs.json')


def make_completion_request(prompt, api_params, api_endpoint, api_headers, max_retries=5):
    """Send a raw-completion request to an OpenAI-compatible /v1/completions endpoint."""
    payload = api_params.copy()
    payload['prompt'] = prompt
    for attempt in range(max_retries):
        try:
            response = requests.post(api_endpoint, json=payload, headers=api_headers)
            response.raise_for_status()
            return [c['text'] for c in response.json()['choices']]
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            sleep(2 ** attempt)
    print("All retry attempts failed.")
    return []


################
# Configurations
################
def get_args():
    parser = argparse.ArgumentParser(description="Instruction Generation Manager (OpenAI-compatible API).")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-4B",
                        help="HF model id used for config lookup / reasoning detection and sent to the API.")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/v1/completions",
                        help="OpenAI-compatible completions endpoint (raw prompt continuation).")
    parser.add_argument("--api_key", type=str, default=None, help="API key (optional for local servers).")

    # Generation Parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=200, help="Number of samples to generate for one time.")
    parser.add_argument("--repeat", type=int, default=None, help="Number of times to repeat the instruction generation. Only available when total prompts is not specified.")
    parser.add_argument("--total_prompts", type=int, default=1000, help="Total number of prompts to generate. If specified, repeat will be ignored.")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--min_instruction_length", type=int, default=10,
                        help="Minimum length (chars) for a generated instruction. "
                             "Shorter/truncated instructions are dropped.")

    # Generation Settings
    parser.add_argument("--early_stopping", action="store_true", default=True, help="Stop generation when the \\n is generated.")
    parser.add_argument("--disable_early_stopping", action="store_false", dest="early_stopping", help="Disable early stopping.")
    parser.add_argument("--system_prompt", action="store_true", help="Enable system prompt for extracting the input.")
    parser.add_argument("--sanitize", action="store_true", help="Sanitize the generated instructions. Only available for Gemma and Llama-3 models.")
    parser.add_argument("--control_tasks", type=str, default=None, choices=[None, "translation", "code", "math"], help="Control tasks for the generation.")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle the outputs returned by the API.")
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle", help="Do not shuffle the outputs returned by the API.")

    # System Settings
    parser.add_argument("--checkpoint_every", type=int, default=100, help="Save checkpoint every n repeats.")
    parser.add_argument("--output_folder", type=str, default="../data")
    parser.add_argument("--job_name", type=str, default=None, help="Job Name. Get from the script.")
    parser.add_argument("--timestamp", type=int, default=int(time.time()), help="Timestamp for the job. Also used as the random seed.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    return parser.parse_args()


def main():
    args = get_args()
    print(f"Instruction Generation Manager. Arguments: {args}")

    if args.sanitize:
        if not ("gemma" in args.model_name.lower() or "llama-3" in args.model_name.lower()):
            raise ValueError("Sanitization is only supported for Gemma and Llama-3 models.")

    if args.total_prompts is None:
        if args.repeat is None:
            raise ValueError("Either total prompts or repeat should be specified.")
        args.total_prompts = args.repeat * args.n
    else:
        args.repeat = int(math.ceil(args.total_prompts / args.n))

    if args.seed is not None:
        random.seed(args.seed)

    # Create output file / folder
    output_filename = f"Magpie_{args.model_name.split('/')[-1]}_{args.total_prompts}_{args.timestamp}_ins.json"
    if not args.job_name:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        output_dir = f"{args.output_folder}/{output_filename}"
    else:
        output_dir = f"{args.output_folder}/{args.job_name}/{output_filename}"
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Obtain config from configs/model_configs.json
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        model_configs = json.load(f)
        if args.model_name not in model_configs:
            raise ValueError(
                f"Model '{args.model_name}' not found in {CONFIG_PATH}. "
                f"Available models: {list(model_configs.keys())}"
            )
        model_config = model_configs[args.model_name]
        if args.control_tasks:
            key = f"pre_query_template_{args.control_tasks}"
            if key not in model_config:
                raise ValueError(f"Config for '{args.model_name}' has no '{key}'. Available keys: {list(model_config.keys())}")
            pre_query_template = model_config[key]
            print(f"Control task: {args.control_tasks}")
        elif args.system_prompt:
            key = "pre_query_template_with_system_prompt"
            if key not in model_config:
                raise ValueError(f"Config for '{args.model_name}' has no '{key}'. Available keys: {list(model_config.keys())}")
            pre_query_template = model_config[key]
            print("System prompt enabled. Warning: The system prompt may degrade the performance.")
        else:
            pre_query_template = model_config["pre_query_template"]
        stop_tokens = model_config["stop_tokens"]
        stop_tokens_assistant = model_config["stop_tokens_assistant"]
        stop_tokens += stop_tokens_assistant

        if args.early_stopping:
            stop_tokens.append("\n")

        print(f"Pre-query template: {pre_query_template}")
        print(f"Stop tokens: {stop_tokens}")

    # API setup
    API_ENDPOINT = args.api_url
    API_HEADERS = {}
    if args.api_key:
        API_HEADERS["Authorization"] = f"Bearer {args.api_key}"
    API_PARAMS = {
        "model": args.model_name,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n": args.n,
        "stop": stop_tokens,
    }

    ################
    # Generate outputs
    ################
    results = []
    for rounds in tqdm(range(args.repeat)):
        output_list = make_completion_request(pre_query_template, API_PARAMS, API_ENDPOINT, API_HEADERS)
        if args.shuffle:
            random.shuffle(output_list)

        for i, completion in enumerate(output_list):
            instruction = completion.strip()
            # Strip reasoning markers (thinking/response) for Qwen3-style models.
            instruction = str_utils.extract_final_answer(instruction, args.model_name)

            # Drop truncated / too-short instructions.
            if len(instruction) < args.min_instruction_length:
                continue

            if args.sanitize:
                sanitized_instruction, class_num = str_utils.instruction_post_process(instruction, args.model_name)
                result = {
                    "id": rounds * args.n + i,
                    "pre_query_template": f"{pre_query_template}",
                    "raw_instruction": instruction,
                    "instruction": sanitized_instruction,
                    "instruction_sanitize_class_num": class_num,
                    "response": None,
                    "created": int(time.time()),
                    "gen_input_configs": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "input_generator": f"{args.model_name}",
                        "seed": args.seed,
                    },
                    "gen_response_configs": None,
                }
            else:
                result = {
                    "id": rounds * args.n + i,
                    "pre_query_template": f"{pre_query_template}",
                    "instruction": instruction,
                    "response": None,
                    "created": int(time.time()),
                    "gen_input_configs": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "input_generator": f"{args.model_name}",
                        "seed": args.seed,
                    },
                    "gen_response_configs": None,
                }
            results.append(result)

        if rounds % args.checkpoint_every == 0:
            with open(output_dir, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Checkpoint saved. Total prompts: {len(results)}")

    with open(output_dir, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Instruction generated from {args.model_name}. Total prompts: {len(results)}")


if __name__ == "__main__":
    main()
