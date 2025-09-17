import json
import os
import random
from collections import defaultdict

import pandas as pd
import wandb
from joblib import Parallel, delayed
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from src.classify_prompt import get_prompt
from src.onlinevllm import OnlineVLLM

random.seed(42)


def get_data():
    def process_data(df):
        df["label"] = df["label"].fillna("none")
        df["label"] = df["label"].astype(str)
        return df

    df = pd.read_csv("data/1a_train.tsv", sep="\t")
    df = process_data(df)

    label2texts = defaultdict(list)
    for label in df["label"].unique():
        label2texts[label.lower()] = df[df["label"] == label]["text"].tolist()

    mn = min(len(texts) for texts in label2texts.values())
    mn = mn // 5 * 5
    for label, texts in label2texts.items():
        label2texts[label] = random.sample(texts, mn)

    dev_data = pd.read_csv("data/1a_dev.tsv", sep="\t")
    dev_data = process_data(dev_data)

    test_data = pd.read_csv("data/1a_test.tsv", sep="\t")

    return label2texts, dev_data, test_data


def run_turn(args, turn_num, label2texts, input_sentence):
    if (turn_num + 1) * args.num_shots > len(label2texts[list(label2texts.keys())[0]]):
        curr_label2texts = {label: random.sample(texts, args.num_shots) for label, texts in label2texts.items()}
    else:
        curr_label2texts = {label: texts[turn_num * args.num_shots : (turn_num + 1) * args.num_shots] for label, texts in label2texts.items()}
    prompt = get_prompt(args, curr_label2texts, input_sentence)
    response = onlinevllm.chat(prompt)
    return response


def evaluate(gold_values, pred_values):
    acc = accuracy_score(gold_values, pred_values)
    precision = precision_score(gold_values, pred_values, average="weighted")
    recall = recall_score(gold_values, pred_values, average="weighted")
    f1 = f1_score(gold_values, pred_values, average="micro")
    return acc, precision, recall, f1


def parse_output(output):
    try:
        if "</think>" in output:
            output = output.split("</think>")[-1]

        # Find the last occurrence of classification tags to handle multiple tags properly
        if "<classification>" in output and "</classification>" in output:
            # Find the last occurrence of classification tags
            start_idx = output.rfind("<classification>")
            end_idx = output.find("</classification>", start_idx)
            if start_idx != -1 and end_idx != -1:
                output = output[start_idx + len("<classification>") : end_idx].strip()
            else:
                # Fallback: try to find any classification content
                parts = output.split("<classification>")
                if len(parts) > 1:
                    output = parts[-1].split("</classification>")[0].strip()

        if output not in ["none", "sexism", "religious hate", "political hate", "profane", "abusive"]:
            # fallback to the first label that appears in the output
            for label in ["sexism", "religious", "political", "profane", "abusive"]:
                if label in output:
                    output = label
                    break
            if output not in ["sexism", "religious", "political", "profane", "abusive"]:
                raise ValueError(f"Invalid Output: {output}")
        return output
    except Exception as e:
        logger.warning(f"$$$$ Failed to parse output: {output}, error: {e}")
        return "none"


import hashlib
import unicodedata


def bengali_hash(text):
    normalized = unicodedata.normalize("NFC", text)
    encoded = normalized.encode("utf-8")
    hash_obj = hashlib.sha256(encoded)
    return hash_obj.hexdigest()


def run_row(args, label2texts, input_sentence):
    hashed_sen = bengali_hash(input_sentence)
    if os.path.exists(f".cache/{hashed_sen}.txt"):
        with open(f".cache/{hashed_sen}.txt", "r") as f:
            prev_input_sentence = f.readline().strip()
            prev_winner = f.readline().strip()
            if prev_input_sentence == input_sentence and prev_winner != "regenerate":
                print(f"Found in cache: {hashed_sen}, skipping...")
                return prev_winner

    copy_label2texts = {label: texts for label, texts in label2texts.items()}

    outputs = Parallel(n_jobs=args.num_turns, backend="threading")(
        delayed(run_turn)(args, turn_num, copy_label2texts, input_sentence) for turn_num in range(args.num_turns)
    )
    outputs = [parse_output(output) for output in outputs]

    max_iterations = 10
    winner = "regenerate"
    while max_iterations > 0:
        counts = {}
        for output in outputs:
            counts[output] = counts.get(output, 0) + 1

        max_count = max(counts.values())
        winners = [key for key, value in counts.items() if value == max_count]

        if len(winners) == 1:
            winner = winners[0]
            break

        # random shuffle the copy_label2texts
        for key in copy_label2texts.keys():
            copy_label2texts[key] = random.sample(copy_label2texts[key], len(copy_label2texts[key]))

        curr_output = run_turn(args, 0, copy_label2texts, input_sentence)
        curr_output = parse_output(curr_output)
        outputs.append(curr_output)
        max_iterations -= 1

    # If we still have a tie after max iterations, return the first winner
    # or "none" as fallback if no winners exist (should not happen)
    # if winners:
    #     winner = winners[0]

    with open(f".cache/{hashed_sen}.txt", "w") as f:
        f.write(input_sentence + "\n")
        f.write(winner + "\n")
    return winner


def get_balanced_test_data(df):
    df["label"] = df["label"].fillna("none")
    df["label"] = df["label"].astype(str)
    df["label"] = df["label"].str.lower()
    path = f"data/1a_{args.split}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    balanced_df = df.groupby("label").sample(n=10, random_state=42)
    balanced_df = balanced_df.reset_index(drop=True)
    balanced_df.to_csv(path, index=False)
    return balanced_df


def main(args):
    print("######################################################################")
    print(json.dumps(args.__dict__, indent=2))
    print("######################################################################")
    label2texts, dev_data, test_data = get_data()
    # assert args.num_shots * args.num_turns <= len(label2texts[list(label2texts.keys())[0]])
    if "balanced" in args.split:
        dev_data = get_balanced_test_data(dev_data)
        test_data = test_data.sample(n=10, random_state=42)

    if "dev" in args.split:
        if os.path.exists(f"output/{args.split}/{args.prompt}/{args.num_shots}_{args.num_turns}_{args.model_id.replace('/', '_')}.csv"):
            print(f"output/{args.split}/{args.prompt}/{args.num_shots}_{args.num_turns}_{args.model_id.replace('/', '_')}.csv already exists")
            return
        pred_labels = []
        gold_labels = []
        func_args = []
        for i, row in dev_data.iterrows():
            input_sentence = row["text"]
            func_args.append((args, label2texts, input_sentence))
            gold_label = row["label"]
            gold_labels.append(gold_label)

        pred_labels = Parallel(n_jobs=-1, backend="threading")(
            delayed(run_row)(args, label2texts, input_sentence) for args, label2texts, input_sentence in tqdm(func_args, desc="Running few-shot inference")
        )
        pred_labels = [pred_label.lower() for pred_label in pred_labels]
        gold_labels = [gold_label.lower() for gold_label in gold_labels]
        acc, precision, recall, f1 = evaluate(gold_labels, pred_labels)

        wandb.init(project="few-shot-hate-speech", name=f"{args.model_id.replace('/', '_')}_{args.num_shots}_{args.num_turns}_{args.prompt}", config=args)
        wandb.log({"accuracy": acc, "precision": precision, "recall": recall, "f1": f1})
        wandb.finish()

        os.makedirs(f"output/{args.split}/{args.prompt}", exist_ok=True)
        with open(f"output/{args.split}/{args.prompt}/{args.num_shots}_{args.num_turns}_{args.model_id.replace('/', '_')}.json", "w") as f:
            json.dump({"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}, f, indent=2)

        dev_data["pred_label"] = pred_labels
        dev_data["gold_label"] = gold_labels
        dev_data = dev_data.reset_index(drop=True)
        dev_data.to_csv(f"output/{args.split}/{args.prompt}/{args.num_shots}_{args.num_turns}_{args.model_id.replace('/', '_')}.csv", index=False)
        print(f"Saved dev data to {f'output/{args.split}/{args.prompt}/{args.num_shots}_{args.num_turns}_{args.model_id.replace("/", "_")}.csv'}")
    else:
        if os.path.exists(f"output/{args.split}/{args.prompt}/{args.num_shots}_{args.num_turns}_{args.model_id.replace('/', '_')}.csv"):
            print(f"output/{args.split}/{args.prompt}/{args.num_shots}_{args.num_turns}_{args.model_id.replace('/', '_')}.csv already exists")
            return
        pred_labels = []
        func_args = []
        for i, row in test_data.iterrows():
            input_sentence = row["text"]
            func_args.append((args, label2texts, input_sentence))

        print(f"Running {len(func_args)} rows")

        pred_labels = Parallel(n_jobs=256, backend="threading")(
            delayed(run_row)(args, label2texts, input_sentence) for args, label2texts, input_sentence in tqdm(func_args, desc="Running few-shot inference")
        )
        pred_labels = [pred_label.lower() for pred_label in pred_labels]

        submission_data = test_data[["id"]]
        submission_data["label"] = pred_labels
        submission_data["model"] = f"{args.model_id.replace('/', '_')}_{args.num_shots}_{args.num_turns}"

        def format_label(label):
            return " ".join(word.capitalize() for word in label.split())

        submission_data["label"] = submission_data["label"].apply(format_label)
        submission_data = submission_data.reset_index(drop=True)
        os.makedirs(f"output/{args.split}/{args.prompt}", exist_ok=True)
        submission_data.to_csv(f"output/{args.split}/{args.prompt}/{args.num_shots}_{args.num_turns}_{args.model_id.replace('/', '_')}.csv", index=False)
        print(f"Saved test data to {f'output/{args.split}/{args.prompt}/{args.num_shots}_{args.num_turns}_{args.model_id.replace("/", "_")}.csv'}")


def submit(args):
    test_data = pd.read_csv(f"data/1a_test.tsv", sep="\t")
    pred_labels = []
    from tqdm import tqdm

    for i, row in tqdm(test_data.iterrows(), desc="Submitting test data"):
        input_sentence = row["text"]
        hashed_sen = bengali_hash(input_sentence)
        # assert os.path.exists(f".cache/{hashed_sen}.txt"), f"Cache file {f'.cache/{hashed_sen}.txt'} not found"
        if os.path.exists(f".cache/{hashed_sen}.txt"):
            with open(f".cache/{hashed_sen}.txt", "r") as f:
                prev_input_sentence = f.readline().strip()
                prev_winner = f.readline().strip()
                if prev_input_sentence == input_sentence and prev_winner != "regenerate":
                    pred_label = prev_winner
                else:
                    raise ValueError(f"Input sentence {input_sentence} not found in cache")
        else:
            with open(f".cache_10_3/{hashed_sen}.txt", "r") as f:
                print("second cache")
                prev_input_sentence = f.readline().strip()
                prev_winner = f.readline().strip()
                if prev_input_sentence == input_sentence and prev_winner != "regenerate":
                    pred_label = prev_winner
                else:
                    raise ValueError(f"Input sentence {input_sentence} not found in cache")

        pred_labels.append(pred_label)

    pred_labels = [pred_label.lower() for pred_label in pred_labels]

    submission_data = test_data[["id"]]
    submission_data["label"] = pred_labels
    submission_data["model"] = f"{args.model_id.replace('/', '_')}_{args.num_shots}_{args.num_turns}"

    def format_label(label):
        return " ".join(word.capitalize() for word in label.split())

    submission_data["label"] = submission_data["label"].apply(format_label)
    submission_data = submission_data.reset_index(drop=True)
    os.makedirs(f"output/{args.split}/{args.prompt}", exist_ok=True)
    submission_data.to_csv(f"output/{args.split}/{args.prompt}/{args.num_shots}_{args.num_turns}_{args.model_id.replace('/', '_')}.csv", index=False)
    print(f"Saved test data to {f'output/{args.split}/{args.prompt}/{args.num_shots}_{args.num_turns}_{args.model_id.replace("/", "_")}.csv'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument("--num_shots", type=int, default=5)
    parser.add_argument("--num_turns", type=int, default=7)
    parser.add_argument("--prompt", type=str, default="classify_with_words")
    parser.add_argument("--split", type=str, default="dev")
    args = parser.parse_args()
    # submit(args)
    onlinevllm = OnlineVLLM(model_id=args.model_id)
    onlinevllm.init_vllm()
    main(args)
    onlinevllm.kill_vllm()
