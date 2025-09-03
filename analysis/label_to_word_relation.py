import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
import re


def analyze_word_label_association(csv_file, text_column, label_column, min_freq=5):
    df = pd.read_csv(csv_file)

    def preprocess_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text

    df["processed_text"] = df[text_column].apply(preprocess_text)

    vectorizer = CountVectorizer(min_df=min_freq, max_df=0.95, stop_words="english", ngram_range=(1, 1))

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
