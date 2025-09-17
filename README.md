# PromptGuard: A Few-Shot Classification Framework Using Majority Voting and Keyword Similarity for Bengali Hate Speech Detection

The project employs a few-shot learning methodology for Bengali hate speech classification. The manager agent orchestrates the process. For each input sentence, it generates a detailed few-shot prompt. These prompts are enriched with a specified number of examples for each of the six hate speech categories, drawn from the training set. Our prompt variation also includes category-specific keywords to guide the model, these keywords are extracted based on the most similar words correlated with the label in the training set.

The classification is performed in multiple "turns". In each turn, a fresh set of few-shot examples is sampled to construct a new prompt, which is then sent to a large language model for inference. The final prediction for a sentence is determined by a majority vote on the classifications from all turns. If a tie occurs, the process iteratively runs additional turns with new, shuffled examples until a single winning category is decided. 

# Bibtex
```
@InProceedings{BLP2025:task1:PromptGuard,
    author = {Hossan, Rakib and Roy Dipta, Shubhashis},
    title = "PromptGuard at BLP-2025 Task 1: A Few-Shot Classification Framework Using Majority Voting and Keyword Similarity for Bengali Hate Speech Detection",
    booktitle = "Proceedings of the 2nd Workshop on Bangla Language Processing (BLP 2025)",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "Association for Computational Linguistics",
}
```
