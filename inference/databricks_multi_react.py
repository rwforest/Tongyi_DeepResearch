"""
Databricks Multi React - Modified version of run_multi_react.py
Uses predict() function instead of VLLM servers
Based on inference/run_multi_react.py
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import threading
from datetime import datetime
import time
import math
from typing import Callable, Optional
from dotenv import load_dotenv

# Import our modified agent
from databricks_react_agent import DatabricksMultiTurnReactAgent


def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()

    # Set default values for missing environment variables
    defaults = {
        'TEMPERATURE': '0.85',
        'PRESENCE_PENALTY': '1.1',
        'ROLLOUT_COUNT': '3',
        'MAX_WORKERS': '30',
        'QWEN_DOC_PARSER_USE_IDP': 'false',
        'NLP_WEB_SEARCH_ONLY_CACHE': 'false',
        'NLP_WEB_SEARCH_ENABLE_READPAGE': 'false',
        'NLP_WEB_SEARCH_ENABLE_SFILTER': 'false',
        'SPECIAL_CODE_MODE': 'false',
    }

    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value

    print("✅ Environment loaded:")
    print(f"   Temperature: {os.environ.get('TEMPERATURE')}")
    print(f"   Presence Penalty: {os.environ.get('PRESENCE_PENALTY')}")
    print(f"   Rollout Count: {os.environ.get('ROLLOUT_COUNT')}")

    # Check for required API keys
    api_keys = ['SERPER_KEY_ID', 'JINA_API_KEYS', 'DASHSCOPE_API_KEY']
    for key in api_keys:
        if os.environ.get(key):
            print(f"   {key}: ✅ Configured")
        else:
            print(f"   {key}: ⚠️ Not configured")


def run_databricks_multi_react(
    predict_function: Callable[[str], str],
    dataset: str = "research_dataset",
    output_path: str = "/tmp/output",
    model_name: str = "databricks-model",
    temperature: float = 0.85,
    top_p: float = 0.95,
    presence_penalty: float = 1.1,
    max_workers: int = 30,
    rollout_count: int = 3,
    total_splits: int = 1,
    worker_split: int = 1,
    data_file: Optional[str] = None
):
    """
    Main function to run multi-turn ReAct inference using predict() function

    Args:
        predict_function: Your predict() function that takes a string and returns response
        dataset: Dataset name
        output_path: Output directory
        model_name: Model identifier
        temperature: Sampling temperature
        top_p: Top-p sampling
        presence_penalty: Presence penalty
        max_workers: Maximum worker threads
        rollout_count: Number of rollouts per question
        total_splits: Total number of data splits
        worker_split: Current worker split (1-indexed)
        data_file: Path to data file (optional)
    """

    print("🚀 Starting Databricks Multi-Turn ReAct Inference")
    print("=" * 60)

    # Validate worker_split
    if worker_split < 1 or worker_split > total_splits:
        print(f"❌ Error: worker_split ({worker_split}) must be between 1 and total_splits ({total_splits})")
        return False

    # Setup output directory
    model_dir = os.path.join(output_path, f"{model_name}_databricks")
    dataset_dir = os.path.join(model_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"📁 Model name: {model_name}")
    print(f"📁 Dataset name: {dataset}")
    print(f"📁 Output directory: {dataset_dir}")
    print(f"📁 Number of rollouts: {rollout_count}")
    print(f"📁 Data splitting: {worker_split}/{total_splits}")

    # Load dataset
    if data_file is None:
        data_file = f"eval_data/{dataset}.jsonl"

    try:
        if data_file.endswith(".json"):
            with open(data_file, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
        elif data_file.endswith(".jsonl"):
            with open(data_file, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")

    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {data_file}")

        # Create a sample dataset for demonstration
        print("📝 Creating sample dataset for demonstration...")
        sample_items = [
            {
                "question": "What are the latest developments in transformer architectures for natural language processing?",
                "answer": "Recent developments include efficient attention mechanisms, mixture of experts, and architectural optimizations."
            },
            {
                "question": "How do attention mechanisms work in neural networks?",
                "answer": "Attention mechanisms allow models to focus on relevant parts of the input sequence when generating outputs."
            },
            {
                "question": "What are the advantages of using large language models for research tasks?",
                "answer": "Large language models can process vast amounts of information, generate coherent text, and assist with complex reasoning tasks."
            }
        ]

        # Save sample dataset
        os.makedirs("eval_data", exist_ok=True)
        with open(data_file, "w", encoding="utf-8") as f:
            for item in sample_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        items = sample_items
        print(f"✅ Created sample dataset with {len(items)} questions at {data_file}")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Error reading or parsing input file {data_file}: {e}")
        return False

    # Apply data splitting
    total_items = len(items)
    items_per_split = math.ceil(total_items / total_splits)
    start_idx = (worker_split - 1) * items_per_split
    end_idx = min(worker_split * items_per_split, total_items)

    items = items[start_idx:end_idx]

    print(f"📊 Total items in dataset: {total_items}")
    print(f"📊 Processing items {start_idx} to {end_idx-1} ({len(items)} items)")

    # Setup output files
    if total_splits > 1:
        output_files = {i: os.path.join(dataset_dir, f"iter{i}_split{worker_split}of{total_splits}.jsonl")
                       for i in range(1, rollout_count + 1)}
    else:
        output_files = {i: os.path.join(dataset_dir, f"iter{i}.jsonl")
                       for i in range(1, rollout_count + 1)}

    # Check for already processed queries
    processed_queries_per_rollout = {}
    for rollout_idx in range(1, rollout_count + 1):
        output_file = output_files[rollout_idx]
        processed_queries = set()

        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "question" in data and "error" not in data:
                                processed_queries.add(data["question"].strip())
                        except json.JSONDecodeError:
                            print(f"⚠️ Warning: Skipping invalid line in output file")
            except FileNotFoundError:
                pass

        processed_queries_per_rollout[rollout_idx] = processed_queries

    # Build task list
    tasks_to_run_all = []
    per_rollout_task_counts = {i: 0 for i in range(1, rollout_count + 1)}

    for rollout_idx in range(1, rollout_count + 1):
        processed_queries = processed_queries_per_rollout[rollout_idx]

        for item in items:
            question = item.get("question", "").strip()

            # Extract question if not present
            if question == "":
                try:
                    if "messages" in item and len(item["messages"]) > 1:
                        user_msg = item["messages"][1]["content"]
                        question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                        item["question"] = question
                except Exception as e:
                    print(f"⚠️ Extract question from user message failed: {e}")

            if not question:
                print(f"⚠️ Warning: Skipping item with empty question")
                continue

            if question not in processed_queries:
                tasks_to_run_all.append({
                    "item": item.copy(),
                    "rollout_idx": rollout_idx,
                })
                per_rollout_task_counts[rollout_idx] += 1

    print(f"📋 Total questions in current split: {len(items)}")
    for rollout_idx in range(1, rollout_count + 1):
        processed = len(processed_queries_per_rollout[rollout_idx])
        to_run = per_rollout_task_counts[rollout_idx]
        print(f"📋 Rollout {rollout_idx}: already processed: {processed}, to run: {to_run}")

    if not tasks_to_run_all:
        print("✅ All rollouts have been completed and no execution is required.")
        return True

    # Initialize ReAct agent
    llm_cfg = {
        'model': model_name,
        'generate_cfg': {
            'max_input_tokens': 320000,
            'max_retries': 10,
            'temperature': temperature,
            'top_p': top_p,
            'presence_penalty': presence_penalty
        },
        'model_type': 'databricks'
    }

    print("🤖 Initializing Databricks ReAct Agent...")
    test_agent = DatabricksMultiTurnReactAgent(
        predict_function=predict_function,
        llm=llm_cfg,
        function_list=["search", "visit", "google_scholar", "PythonInterpreter"]
    )

    # Setup thread locks for writing
    write_locks = {i: threading.Lock() for i in range(1, rollout_count + 1)}

    print(f"🚀 Starting processing with {max_workers} workers...")
    print(f"📝 Total tasks to process: {len(tasks_to_run_all)}")

    # Process tasks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(test_agent._run, task, model_name): task
            for task in tasks_to_run_all
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run_all), desc="Processing All Rollouts"):
            task_info = future_to_task[future]
            rollout_idx = task_info["rollout_idx"]
            output_file = output_files[rollout_idx]

            try:
                result = future.result()

                with write_locks[rollout_idx]:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

            except concurrent.futures.TimeoutError:
                question = task_info["item"].get("question", "")
                print(f'⏰ Timeout (>1800s): "{question}" (Rollout {rollout_idx})')
                future.cancel()

                error_result = {
                    "question": question,
                    "answer": task_info["item"].get("answer", ""),
                    "rollout_idx": rollout_idx,
                    "rollout_id": rollout_idx,
                    "error": "Timeout (>1800s)",
                    "messages": [],
                    "prediction": "[Failed]"
                }

                with write_locks[rollout_idx]:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

            except Exception as exc:
                question = task_info["item"].get("question", "")
                print(f'❌ Task for question "{question}" (Rollout {rollout_idx}) generated an exception: {exc}')

                error_result = {
                    "question": question,
                    "answer": task_info["item"].get("answer", ""),
                    "rollout_idx": rollout_idx,
                    "rollout_id": rollout_idx,
                    "error": f"Future resolution failed: {exc}",
                    "messages": [],
                    "prediction": "[Failed]",
                }

                with write_locks[rollout_idx]:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

    print("✅ All tasks completed!")
    print(f"✅ All {rollout_count} rollouts completed!")

    # Print output file locations
    print(f"📁 Results saved to:")
    for rollout_idx, file_path in output_files.items():
        if os.path.exists(file_path):
            print(f"   Rollout {rollout_idx}: {file_path}")

    return True


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Databricks Multi-Turn ReAct Inference")

    parser.add_argument("--dataset", type=str, default="research_dataset", help="Dataset name")
    parser.add_argument("--output", type=str, default="/tmp/databricks_react_output", help="Output directory")
    parser.add_argument("--model", type=str, default="databricks-model", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.85, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p")
    parser.add_argument("--presence_penalty", type=float, default=1.1, help="Presence penalty")
    parser.add_argument("--max_workers", type=int, default=30, help="Max workers")
    parser.add_argument("--roll_out_count", type=int, default=3, help="Rollout count")
    parser.add_argument("--total_splits", type=int, default=1, help="Total splits")
    parser.add_argument("--worker_split", type=int, default=1, help="Worker split")
    parser.add_argument("--data_file", type=str, help="Data file path")

    args = parser.parse_args()

    print("🚀 Databricks Multi-Turn ReAct Inference")
    print("=" * 50)

    # Load environment
    load_environment()

    print("⚠️  This script requires integration with your predict() function.")
    print("Please modify the script to pass your actual predict function.")
    print()
    print("Example integration:")
    print("""
    # Your existing predict function
    def your_predict_function(prompt: str) -> str:
        # Your implementation here
        return model_response

    # Run the inference
    success = run_databricks_multi_react(
        predict_function=your_predict_function,
        dataset=args.dataset,
        output_path=args.output,
        model_name=args.model,
        temperature=args.temperature,
        # ... other parameters
    )
    """)


if __name__ == "__main__":
    main()