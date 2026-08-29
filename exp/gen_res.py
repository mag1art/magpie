import os
import sys
import argparse
import json
import requests
import concurrent.futures
from time import sleep
from tqdm import tqdm
from utils import load_dataset_from_file, save_dataset, make_api_request_with_retry
import str_utils

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

args = get_args()
print(f"Response Generation Manager. Arguments: {args}")

if args.input_file is None:
    raise ValueError("Please specify the input file path.")

MODEL_NAME = args.model_name
INPUT_FILE_NAME = args.input_file
BATCH_SIZE = args.batch_size
CHECKPOINT_FILE = f"{INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]}_res_checkpoint.json"
CHECKPOINT_EVERY = args.checkpoint_every
SAVED_FILE = f"{INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]}_res.json"

# Obtain config from configs/model_configs.json
with open("../configs/model_configs.json", "r") as f:
    model_configs = json.load(f)
    model_config = model_configs[args.model_name]
    stop_tokens = model_config["stop_tokens"]

# API Setup (OpenAI-compatible chat completions endpoint)
API_ENDPOINT = args.api_url
API_HEADERS = {}
if args.api_key:
    API_HEADERS["Authorization"] = f"Bearer {args.api_key}"
API_PARAMS = {
    "model": MODEL_NAME,
    "max_tokens": args.max_tokens,
    "temperature": args.temperature,
    "top_p": args.top_p,
    "stop": stop_tokens
}
# repetition_penalty is not a standard OpenAI param; only send it when != 1.0
# (vLLM accepts it, but some OpenAI-compatible servers reject unknown params).
if args.repetition_penalty != 1.0:
    API_PARAMS["repetition_penalty"] = args.repetition_penalty


# Process a batch of data using the API
def process_batch_with_api(batch):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_item = {
            executor.submit(
                make_api_request_with_retry,
                [{'content': item['instruction'], 'role': 'user'}],
                API_PARAMS,
                API_ENDPOINT,
                API_HEADERS,
            ): item
            for item in batch
        }

        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                api_response = future.result()
                item['response'] = str_utils.extract_final_answer(api_response, MODEL_NAME)
                item['gen_response_configs'] = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "max_tokens": args.max_tokens,
                    "stop_tokens": stop_tokens,
                    "output_generator": MODEL_NAME,
                    "engine": "api",
                }
            except Exception as e:
                print(f"Failed to process item: {item} with error: {str(e)}")
                item['response'] = ""

    return batch


# Generate outputs, update dataset in batches, and overwrite checkpoint
def generate_and_update(dataset):
    if os.path.exists(CHECKPOINT_FILE):
        last_checkpoint_idx = len(load_dataset_from_file(CHECKPOINT_FILE))
        print(f"Checkpoint file found. Resuming from last checkpoint with index {last_checkpoint_idx}.")
        dataset[:last_checkpoint_idx] = load_dataset_from_file(CHECKPOINT_FILE)
        num_batches = (len(dataset) - last_checkpoint_idx + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Remaining number of batches: {num_batches}")
    else:
        last_checkpoint_idx = 0
        num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Total number of batches: {num_batches}")

    for i in tqdm(range(num_batches)):
        start_idx = i * BATCH_SIZE + last_checkpoint_idx
        end_idx = min((i + 1) * BATCH_SIZE + last_checkpoint_idx, len(dataset))
        batch = dataset[start_idx:end_idx]
        batch = process_batch_with_api(batch)

        dataset[start_idx:end_idx] = batch
        if i % CHECKPOINT_EVERY == 0:
            save_dataset(dataset[:end_idx], CHECKPOINT_FILE)
            print(f"Dataset checkpoint saved after batch {i + 1}.")

    return dataset


def main():
    dataset = load_dataset_from_file(INPUT_FILE_NAME)
    print("Start OpenAI-compatible API engine...")
    updated_dataset = generate_and_update(dataset)

    save_dataset(updated_dataset, SAVED_FILE)
    os.remove(CHECKPOINT_FILE)
    print("Final dataset saved. Checkpoint removed.")


if __name__ == "__main__":
    main()
