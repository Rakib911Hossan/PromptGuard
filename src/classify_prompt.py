classify_prompt = """\
Your task is to classify a Bengali sentence into one of six hate speech categories: none, sexism, abusive, profane, religious hate, or political hate. Here are examples for each category:

<examples>
{EXAMPLES}
</examples>

Now, classify the following Bengali sentence into one of the above categories:

<input_sentence>
{INPUT_SENTENCE}
</input_sentence>

Analyze the sentence carefully, comparing it to the examples provided. Consider the tone, words used, and overall context of the sentence. Determine which category it most closely aligns with.

Your output should be a single word representing the category classification. Use only one of these exact category names: none, abusive, profane, religious hate, or political hate.

Provide your classification inside <classification> tags."""


def get_prompt(args, label2texts, input_sentence):
    """
    Generate a few-shot classification prompt for hate speech detection.

    This function creates a structured prompt that includes examples for each
    hate speech category and the input sentence to be classified. The prompt
    follows a specific template designed for Bengali hate speech classification.

    Args:
        args: Command line arguments containing num_shots parameter
        label2texts (dict): Dictionary mapping labels to lists of example texts
        input_sentence (str): The Bengali sentence to be classified

    Returns:
        str: Formatted prompt string ready for language model inference
    """
    examples = ""
    for label, texts in label2texts.items():
        curr_examples = "\n".join(texts[: args.num_shots])
        examples += f"<{label}>\n"
        examples += curr_examples
        examples += f"\n</{label}>\n"
    return classify_prompt.format(EXAMPLES=examples, INPUT_SENTENCE=input_sentence)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shots", type=int, default=5)
    args = parser.parse_args()
    label2texts = {
        "none": ["আমি বাংলা ভাষার জন্য একটি ভাষা শিখছি"],
        "abusive": ["আমি বাংলা ভাষার জন্য একটি ভাষা শিখছি"],
        "profane": ["আমি বাংলা ভাষার জন্য একটি ভাষা শিখছি"],
        "religious hate": ["আমি বাংলা ভাষার জন্য একটি ভাষা শিখছি"],
        "political hate": ["আমি বাংলা ভাষার জন্য একটি ভাষা শিখছি"],
    }
    print(get_prompt(args, label2texts, "আমি বাংলা ভাষার জন্য একটি ভাষা শিখছি"))
