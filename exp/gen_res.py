import os
import sys
import argparse
import json
import concurrent.futures
from tqdm import tqdm
from utils import load_dataset_from_file, save_dataset, make_api_request_with_retry
import str_utils

# Resolve the config path relative to this file so the script works from any CWD.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'model_configs.json')


################
# Configurations
################
def get_args():
    parser = argparse.ArgumentParser(description="Response Generation Manager (OpenAI-compatible API).")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-4B",
                        help="HF model id used for config lookup / reasoning detection and sent to the API.")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples per batch")
    parser.add_argument("--checkpoint_every", type=int, default=20, help="Save checkpoint every n batches")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/v1/chat/completions",
                        help="OpenAI-compatible chat completions endpoint.")
    parser.add_argument("--api_key", type=str, default=None, help="API key (optional for local servers).")

    # Generation Parameters
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    return parser.parse_args()


def load_model_config(model_name):
    """Load a model's chat template + stop tokens from configs/model_configs.json."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        model_configs = json.load(f)
    if model_name not in model_configs:
        raise ValueError(
            f"Model '{model_name}' not found in {CONFIG_PATH}. "
            f"Available models: {list(model_configs.keys())}"
        )
    return model_configs[model_name]


def build_api_params(args, stop_tokens):
    """Build the OpenAI-compatible chat/completions payload."""
    params = {
        "model": args.model_name,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stop": stop_tokens,
    }
    # repetition_penalty is not a standard OpenAI param; only send it when != 1.0
    # (vLLM accepts it, but some OpenAI-compatible servers reject unknown params).
    if args.repetition_penalty != 1.0:
        params["repetition_penalty"] = args.repetition_penalty
    return params


# Process a batch of data using the API.
# HTTP calls are I/O-bound, so a thread pool is more efficient than processes.
def process_batch_with_api(batch, api_params, api_endpoint, api_headers, model_name, gen_config):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_item = {
            executor.submit(
                make_api_request_with_retry,
                [{'content': item['instruction'], 'role': 'user'}],
                api_params,
                api_endpoint,
                api_headers,
            ): item
            for item in batch
        }

        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                api_response = future.result()
                item['response'] = str_utils.extract_final_answer(api_response, model_name) or ""
                item['gen_response_configs'] = gen_config
            except Exception as e:
                print(f"Failed to process item: {item} with error: {str(e)}")
                item['response'] = ""

    return batch


# Generate outputs, update dataset in batches, and overwrite checkpoint
def generate_and_update(dataset, batch_size, checkpoint_file, checkpoint_every,
                        api_params, api_endpoint, api_headers, model_name, gen_config):
    if os.path.exists(checkpoint_file):
        last_checkpoint_idx = len(load_dataset_from_file(checkpoint_file))
        print(f"Checkpoint file found. Resuming from last checkpoint with index {last_checkpoint_idx}.")
        dataset[:last_checkpoint_idx] = load_dataset_from_file(checkpoint_file)
        num_batches = (len(dataset) - last_checkpoint_idx + batch_size - 1) // batch_size
        print(f"Remaining number of batches: {num_batches}")
    else:
        last_checkpoint_idx = 0
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        print(f"Total number of batches: {num_batches}")

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size + last_checkpoint_idx
        end_idx = min((i + 1) * batch_size + last_checkpoint_idx, len(dataset))
        batch = dataset[start_idx:end_idx]
        batch = process_batch_with_api(batch, api_params, api_endpoint, api_headers, model_name, gen_config)

        dataset[start_idx:end_idx] = batch
        if i % checkpoint_every == 0:
            save_dataset(dataset[:end_idx], checkpoint_file)
            print(f"Dataset checkpoint saved after batch {i + 1}.")

    return dataset


def main():
    args = get_args()
    print(f"Response Generation Manager. Arguments: {args}")

    if args.input_file is None:
        raise ValueError("Please specify the input file path.")

    model_config = load_model_config(args.model_name)
    stop_tokens = model_config["stop_tokens"]

    # API Setup (OpenAI-compatible chat completions endpoint)
    api_endpoint = args.api_url
    api_headers = {}
    if args.api_key:
        api_headers["Authorization"] = f"Bearer {args.api_key}"
    api_params = build_api_params(args, stop_tokens)

    gen_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_tokens,
        "stop_tokens": stop_tokens,
        "output_generator": args.model_name,
        "engine": "api",
    }

    input_file = args.input_file
    checkpoint_file = f"{input_file[:input_file.rfind('.')]}_res_checkpoint.json"
    saved_file = f"{input_file[:input_file.rfind('.')]}_res.json"

    dataset = load_dataset_from_file(input_file)
    print("Start OpenAI-compatible API engine...")
    updated_dataset = generate_and_update(
        dataset,
        args.batch_size,
        checkpoint_file,
        args.checkpoint_every,
        api_params,
        api_endpoint,
        api_headers,
        args.model_name,
        gen_config,
    )

    save_dataset(updated_dataset, saved_file)
    os.remove(checkpoint_file)
    print("Final dataset saved. Checkpoint removed.")


if __name__ == "__main__":
    main()
