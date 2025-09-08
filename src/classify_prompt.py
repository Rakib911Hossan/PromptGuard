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

Your final output should be the category classification. Use only one of these exact category names: none, sexism, abusive, profane, religious hate, or political hate.

Provide your classification inside <classification> tags."""

improved_classify_prompt = """\
You are an AI language model specialized in detecting hate speech in Bengali. Your task is to classify a given Bengali sentence into one of six categories: none, sexism, abusive, profane, religious hate, or political hate.

First, review these examples of sentences for each category:

<examples>
{EXAMPLES}
</examples>

Here's the Bengali sentence you need to classify:

<input_sentence>
{INPUT_SENTENCE}
</input_sentence>

Before making your final classification, analyze the sentence in detail. Consider the following:
1. Compare the sentence to the provided examples.
2. Examine the tone, specific words used, and overall context.
3. For each category (none, sexism, abusive, profane, religious hate, political hate):
   - List evidence for classifying the sentence into this category.
   - List evidence against classifying the sentence into this category.
4. Summarize your findings and explain your final decision.

Conduct your analysis inside <category_analysis> tags. It's OK for this section to be quite long.

After your analysis, classify the sentence into one of the six categories. Your final output should be the category name within <classification> tags.

Example output format:
<category_analysis>
[Your detailed analysis here]
</category_analysis>

<classification>category_name</classification>

Remember to use only one of these exact category names: none, sexism, abusive, profane, religious hate, or political hate.
"""

classify_with_words_prompt = """\
You are an AI language model specialized in detecting hate speech in Bengali. Your task is to classify a given Bengali sentence into one of six categories: none, sexism, abusive, profane, religious hate, or political hate.

First, review these examples of sentences for each category:

<examples>
{EXAMPLES}
</examples>

Now, consider these common words associated with each category. Note that the presence of these words doesn't guarantee classification into that category, but they can be helpful indicators:

<category_keywords>
abusive: দালাল, টিভি, ফালতু, চোর, মিথ্যা, পাগল, জুতা, লজ্জা, আমিন
profane: বাল, মাগি, খানকি, বেশ্যা, দফা, বাচ্চা, সালা, শালা, মাদারচোদ, কুত্তা, জারজ, পোলা, শুয়োর
religious hate: মুসলিম, হিন্দু, ইহুদি, মুসলমান, গজব, ধর্ম, ইসলাম, কাফের, মসজিদ, ধর্মীয়, মোল্লা, আল্লাহ
political hate: ভোট, বিএনপি, আওয়ামী, লীগ, সরকার, নির্বাচন, হাসিনা, অবৈধ, জনগণ, পার্টি, দল, চোর, রাজনীতি
sexism: নারী, পরকিয়া, মহিলা, পুরুষ, হিজরা, বিয়ে, লিঙ্গ, হোটেল, মেয়ে, বেডা, আবাসিক
</category_keywords>

Here's the Bengali sentence you need to classify:

<input_sentence>
{INPUT_SENTENCE}
</input_sentence>

Before making your final classification, analyze the sentence in detail. Consider the following:
1. Compare the sentence to the provided examples.
2. Examine the tone, specific words used, and overall context.
3. Check if any words from the category_keywords are present and relevant.
4. For each category (none, sexism, abusive, profane, religious hate, political hate):
   - List evidence for classifying the sentence into this category.
   - List evidence against classifying the sentence into this category.
5. Summarize your findings and explain your final decision.

Conduct your analysis inside <category_analysis> tags. It's OK for this section to be quite long.

After your analysis, classify the sentence into one of the six categories. Your final output should be the category name within <classification> tags.

Example output format:
<category_analysis>
[Your detailed analysis here]
</category_analysis>

<classification>category_name</classification>

Remember to use only one of these exact category names: none, sexism, abusive, profane, religious hate, or political hate.
"""


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
        curr_examples = ""
        for text in texts[: args.num_shots]:
            curr_examples += f"- {text}\n"
        examples += f"<{label}>\n"
        examples += curr_examples
        examples += f"</{label}>\n\n"
    examples = examples.strip()
    examples = f"\n{examples}\n"
    return improved_classify_prompt.format(EXAMPLES=examples, INPUT_SENTENCE=input_sentence)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shots", type=int, default=5)
    args = parser.parse_args()
    label2texts = {
        "none": ["আমি বাংলা ভাষার জন্য একটি ভাষা শিখছি", "আজ আবহাওয়া খুব ভালো", "আমি বই পড়তে পছন্দ করি"],
        "abusive": ["তুমি একটা বোকা", "তোমার বুদ্ধি নেই", "তুমি অকর্মা"],
        "profane": ["তোমার মা...", "গাধার বাচ্চা", "পাগলের বাচ্চা"],
        "religious hate": ["হিন্দুদের সবাই মূর্খ", "মুসলমানরা সব খারাপ", "খ্রিস্টানরা মিথ্যাবাদী"],
        "political hate": ["রাজনীতিবিদরা সব চোর", "সরকার সবাই দুর্নীতিবাজ", "নেতারা সব মিথ্যাবাদী"],
        "sexism": ["মেয়েরা শুধু রান্না করতে পারে", "নারীদের বুদ্ধি কম", "মেয়েরা শুধু সাজগোজ করে"],
    }
    print(get_prompt(args, label2texts, "আমি বাংলা ভাষার জন্য একটি ভাষা শিখছি"))
