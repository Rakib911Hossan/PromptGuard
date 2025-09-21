import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from collections import defaultdict
import re


def analyze_word_label_association(csv_file, text_column, label_column, min_freq=5):
    """
    Analyze the statistical association between words and labels using chi-square test.

    This function performs chi-square feature selection to identify words that are
    significantly associated with specific hate speech labels. It preprocesses Bengali
    text by keeping only Bengali characters, English letters, and whitespace.

    Args:
        csv_file (str): Path to the CSV file containing the dataset
        text_column (str): Name of the column containing text data
        label_column (str): Name of the column containing labels
        min_freq (int, optional): Minimum document frequency for words. Defaults to 5.

    Returns:
        dict: Dictionary mapping labels to lists of (word, chi2_score, p_value) tuples
              sorted by chi-square score in descending order
    """
    df = pd.read_csv(csv_file, sep="\t")
    # cast label to string
    df["label"] = df["label"].astype(str)
    print(df.head())

    def preprocess_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Keep Bengali characters, English letters, and whitespace
        text = re.sub(r"[^a-zA-Z\u0980-\u09FF\s]", "", text)
        return text

    df["processed_text"] = df[text_column].apply(preprocess_text)

    vectorizer = CountVectorizer(min_df=min_freq, max_df=0.95, ngram_range=(1, 1), token_pattern=r"[\u0980-\u09FF]+")

    X = vectorizer.fit_transform(df["processed_text"])
    feature_names = vectorizer.get_feature_names_out()

    results = {}
    unique_labels = df[label_column].unique()

    for label in unique_labels:
        y = (df[label_column] == label).astype(int)

        chi2_scores, p_values = chi2(X, y)

        word_scores = list(zip(feature_names, chi2_scores, p_values))
        word_scores = [(word, score, p_val) for word, score, p_val in word_scores if p_val < 0.05]
        word_scores.sort(key=lambda x: x[1], reverse=True)

        results[label] = word_scores[:20]

    return results


def print_results(results):
    """
    Print the word-label association analysis results in a formatted manner.

    This function displays the top 20 words most associated with each label,
    showing their chi-square scores and p-values in a readable format.

    Args:
        results (dict): Dictionary containing word-label association results
                       from analyze_word_label_association function
    """
    for label, word_scores in results.items():
        print(f"\nTop 20 words most associated with '{label}':")
        print("-" * 50)
        for i, (word, score, p_val) in enumerate(word_scores, 1):
            print(f"{i:2d}. {word:<15} (χ²={score:.2f}, p={p_val:.4f})")


if __name__ == "__main__":
    csv_file = "data/1a_train.tsv"
    text_col = "text"
    label_col = "label"

    results = analyze_word_label_association(csv_file, text_col, label_col)
    print_results(results)
