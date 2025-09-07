import random
from collections import defaultdict

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from src.classify_prompt import get_prompt
from src.onlinevllm import OnlineVLLM

random.seed(42)


def get_data():
    """
    Load and preprocess training and development data for hate speech classification.

    This function loads the training data, processes it by filling missing labels and
    converting to string format, creates balanced examples for each label, and loads
    the development/test data for evaluation.

    Returns:
        tuple: A tuple containing:
            - label2texts (dict): Dictionary mapping labels to lists of example texts
            - dev_data (pd.DataFrame): Development dataset for evaluation
            - test_data (None): Test data (currently not used)
    """

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

    test_data = pd.read_csv("data/1a_dev_test.tsv", sep="\t")

    return label2texts, dev_data, test_data


def run_turn(args, turn_num, label2texts, input_sentence):
    """
    Execute a single turn of few-shot inference for hate speech classification.

    This function selects the appropriate examples for the current turn, generates
    a prompt using the selected examples, and gets a response from the language model.

    Args:
        args: Command line arguments containing num_shots parameter
        turn_num (int): Current turn number (0-indexed)
        label2texts (dict): Dictionary mapping labels to lists of example texts
        input_sentence (str): The sentence to be classified

    Returns:
        str: The raw response content from the language model
    """
    curr_label2texts = {label: texts[turn_num * args.num_shots : (turn_num + 1) * args.num_shots] for label, texts in label2texts.items()}
    prompt = get_prompt(args, curr_label2texts, input_sentence)
    response = onlinevllm.chat(prompt)
    return response.choices[0].message.content


def evaluate(gold_values, pred_values):
    """
    Calculate evaluation metrics for hate speech classification.

    Computes accuracy, precision, recall, and F1 score to assess the performance
    of the classification model.

    Args:
        gold_values (list): List of ground truth labels
        pred_values (list): List of predicted labels

    Returns:
        tuple: A tuple containing (accuracy, precision, recall, f1_score)
    """
    acc = accuracy_score(gold_values, pred_values)
    precision = precision_score(gold_values, pred_values, average="weighted")
    recall = recall_score(gold_values, pred_values, average="weighted")
    f1 = f1_score(gold_values, pred_values, average="micro")
    return acc, precision, recall, f1


def parse_output(output):
    """
    Parse the raw model output to extract the classification result.

    This function extracts the classification from the model's response by looking
    for content within <classification> tags. If parsing fails, it returns "none".

    Args:
        output (str): Raw response from the language model

    Returns:
        str: Extracted classification label or "none" if parsing fails
    """
    try:
        if "</think>" in output:
            output = output.split("</think>")[-1]
        while "<classification>" in output and "</classification>" in output:
            output = output.split("<classification>")[1].split("</classification>")[0].strip()
        return output
    except Exception as e:
        logger.warning(f"$$$$ Failed to parse output: {output}, error: {e}")
        return "none"


def run_row(args, label2texts, input_sentence):
    """
    Run multiple turns of few-shot inference with majority voting for final classification.

    This function executes multiple turns of classification and uses majority voting
    to determine the final prediction. If there's a tie, it runs additional turns
    with shuffled examples until a clear winner emerges or max iterations are reached.

    Args:
        args: Command line arguments containing num_turns parameter
        label2texts (dict): Dictionary mapping labels to lists of example texts
        input_sentence (str): The sentence to be classified

    Returns:
        str: Final classification label determined by majority voting
    """
    copy_label2texts = {label: texts for label, texts in label2texts.items()}

    outputs = []
    for turn_num in range(args.num_turns):
        output = run_turn(args, turn_num, copy_label2texts, input_sentence)
        output = parse_output(output)
        outputs.append(output)

    max_iterations = 10
    while max_iterations > 0:
        counts = {}
        for output in outputs:
            counts[output] = counts.get(output, 0) + 1

        max_count = max(counts.values())
        winners = [key for key, value in counts.items() if value == max_count]

        if len(winners) == 1:
            return winners[0]

        # random shuffle the copy_label2texts
        for key in copy_label2texts.keys():
            copy_label2texts[key] = random.sample(copy_label2texts[key], len(copy_label2texts[key]))

        curr_output = run_turn(args, 0, copy_label2texts, input_sentence)
        curr_output = parse_output(curr_output)
        outputs.append(curr_output)
        max_iterations -= 1

    return winners[0]


def get_balanced_test_data(df):
    """
    Create a balanced subset of the test data for debugging purposes.

    This function processes the dataframe by filling missing labels, converting to
    lowercase, and sampling exactly 2 examples per label to create a balanced
    dataset for faster debugging.

    Args:
        df (pd.DataFrame): Input dataframe with text and label columns

    Returns:
        pd.DataFrame: Balanced dataframe with 2 samples per label
    """
    df["label"] = df["label"].fillna("none")
    df["label"] = df["label"].astype(str)
    df["label"] = df["label"].str.lower()
    balanced_df = df.groupby("label").sample(n=2, random_state=42)
    balanced_df = balanced_df.reset_index(drop=True)
    return balanced_df


def main(args):
    """
    Main function to run few-shot hate speech classification evaluation.

    This function orchestrates the entire evaluation pipeline: loads data, runs
    parallel inference on the development set, evaluates performance metrics,
    and saves results to a CSV file.

    Args:
        args: Command line arguments containing model configuration and parameters
    """
    label2texts, dev_data, test_data = get_data()
    assert args.num_shots * args.num_turns <= len(label2texts[list(label2texts.keys())[0]])
    if args.debug:
        dev_data = get_balanced_test_data(dev_data)
        test_data = test_data.sample(n=10, random_state=42)

    if not args.test:
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
        print(pred_labels)
        pred_labels = [pred_label.lower() for pred_label in pred_labels]
        gold_labels = [gold_label.lower() for gold_label in gold_labels]
        acc, precision, recall, f1 = evaluate(gold_labels, pred_labels)
        print(f"Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")

        dev_data["pred_label"] = pred_labels
        dev_data["gold_label"] = gold_labels
        dev_data = dev_data.reset_index(drop=True)
        dev_data.to_csv(f"output/dev/{args.num_shots}_{args.num_turns}_{args.model_id.replace('/', '_')}.csv", index=False)
    else:
        pred_labels = []
        func_args = []
        for i, row in test_data.iterrows():
            input_sentence = row["text"]
            func_args.append((args, label2texts, input_sentence))

        pred_labels = Parallel(n_jobs=-1, backend="threading")(
            delayed(run_row)(args, label2texts, input_sentence) for args, label2texts, input_sentence in tqdm(func_args, desc="Running few-shot inference")
        )
        pred_labels = [pred_label.lower() for pred_label in pred_labels]

        submission_data = test_data[["id"]]
        submission_data["label"] = pred_labels
        submission_data["model"] = args.model_id

        def format_label(label):
            return " ".join(word.capitalize() for word in label.split())

        submission_data["label"] = submission_data["label"].apply(format_label)
        submission_data = submission_data.reset_index(drop=True)
        submission_data.to_csv(f"output/test/{args.num_shots}_{args.num_turns}_{args.model_id.replace('/', '_')}.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument("--num_shots", type=int, default=5)
    parser.add_argument("--num_turns", type=int, default=7)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()
    onlinevllm = OnlineVLLM(model_id=args.model_id)
    onlinevllm.init_vllm()
    main(args)
    onlinevllm.kill_vllm()
