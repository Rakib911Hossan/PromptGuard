#!/usr/bin/env python3
"""
Simple Python conversion of test.sh
"""

import os

MODELS = [
    # "Qwen/Qwen3-30B-A3B-Thinking-2507",
    # "Qwen/Qwen3-30B-A3B-Instruct-2507",
    # "Qwen/Qwen3-4B-Thinking-2507",
    # "Qwen/Qwen3-4B-Instruct-2507",
    # "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
    # "Qwen/Qwen3-14B",
    # "Qwen/Qwen3-8B",
    # "Qwen/Qwen3-4B",
    # "Qwen/Qwen3-1.7B",
    # "Qwen/Qwen3-0.6B",
    # "Qwen/QwQ-32B",
    # "openai/gpt-oss-20b",
    # "openai/gpt-oss-120b",
]

NUM_SHOTS = [
    3,
    # 7,
    # 10,
    # 16,
    # 20,
    # 30,
]
NUM_TURNS = [
    3,
    7,
    10,
    16,
]

SPLITS = [
    # "dev",
    # "dev_balanced",
    # "test",
    "dev_test_labeled",
    # "dev_test_labeled_balanced",
]

PROMPTS = [
    # "classify",
    # "improved_classify",
    "classify_with_words",
]


def run_command(cmd):
    print(cmd)
    os.system(cmd)


def get_cmd(model_id, num_turns, num_shots, split, prompt, log_file):
    cmd = f"""PYTHONPATH=. uv run src/few_shot.py --model_id {model_id} --num_turns {num_turns} --num_shots {num_shots} --split {split} --prompt {prompt}"""
    cmd += f" | tee {log_file}"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    return cmd


def main():
    print("=" * 50)
    print("RUNNING DEVELOPMENT EXPERIMENTS")
    print("=" * 50)

    for prompt in PROMPTS:
        for i in NUM_TURNS:
            for j in NUM_SHOTS:
                # if i * j > 120:
                #     print(f"Skipping {i} turns and {j} shots because it exceeds 120")
                #     continue
                for model_id in MODELS:
                    for split in SPLITS:
                        if os.path.exists(f"output/{split}/{prompt}/{j}_{i}_{model_id.replace('/', '_')}.json"):
                            print(f"Skipping {i} turns and {j} shots and {model_id} and {split} because it already exists")
                            continue
                        print(f"Running for {i} turns and {j} shots and {model_id} and {split}, with prompt {prompt}")
                        cmd = get_cmd(model_id, i, j, split, prompt, f"logs/{split}/{prompt}/few_shot_submission_{model_id.replace('/', '_')}_{j}-{i}.log")
                        run_command(cmd)


if __name__ == "__main__":
    main()
