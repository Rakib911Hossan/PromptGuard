import random
from joblib import Parallel, delayed
from collections import defaultdict

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

    # test_data = pd.read_csv("data/1a_dev.tsv", sep="\t")
    # test_data = process_data(test_data)
    test_data = None
    dev_data = pd.read_csv("data/1a_dev_test.tsv", sep="\t")
    dev_data = process_data(dev_data)

    return label2texts, dev_data, test_data


def run_turn(args, turn_num, label2texts, input_sentence):
    curr_label2texts = {label: texts[turn_num * args.num_shots : (turn_num + 1) * args.num_shots] for label, texts in label2texts.items()}
    prompt = get_prompt(args, curr_label2texts, input_sentence)
    response = onlinevllm.chat(prompt)
    return response.choices[0].message.content


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
        return output.split("<classification>")[1].split("</classification>")[0]
    except:
        return "none"


def run_row(args, label2texts, input_sentence):
    copy_label2texts = {label: texts for label, texts in label2texts.items()}

    outputs = []
    for turn_num in range(args.num_turns):
        output = run_turn(args, turn_num, copy_label2texts, input_sentence)
        output = parse_output(output)
        outputs.append(output)

    while True:
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

        outputs.append(run_turn(args, 0, copy_label2texts, input_sentence))


def get_balanced_test_data(df):
    df["label"] = df["label"].fillna("none")
    df["label"] = df["label"].astype(str)
    df["label"] = df["label"].str.lower()
    balanced_df = df.groupby("label").sample(n=2, random_state=42)
    balanced_df = balanced_df.reset_index(drop=True)
    return balanced_df


def main(args):
    label2texts, dev_data, test_data = get_data()
    assert args.num_shots * args.num_turns <= len(label2texts[list(label2texts.keys())[0]])
    if args.debug:
        dev_data = get_balanced_test_data(dev_data)

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
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    dev_data["pred_label"] = pred_labels
    dev_data["gold_label"] = gold_labels
    dev_data = dev_data.reset_index(drop=True)
    dev_data.to_csv(f"data/few_shot_results_{args.num_shots}_{args.num_turns}.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--num_shots", type=int, default=5)
    parser.add_argument("--num_turns", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    onlinevllm = OnlineVLLM(model_id=args.model_id)
    onlinevllm.init_vllm()
    main(args)
