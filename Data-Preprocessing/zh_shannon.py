import numpy as np
from collections import Counter
import re
import jieba
import pandas as pd

# Use jieba for Chinese word segmentation
def tokenize_chinese(text):
    # Segment text and remove extra spaces
    return list(jieba.cut(text))

# Compute significance and Shannon entropy for words
def compute_significance_and_entropy(ner_words, word_counts, total_words):
    significance_entropy = {}
    for word in ner_words:
        # Calculate term frequency (TF)
        word_freq = word_counts[word] if word in word_counts else 1
        significance_level = word_freq / total_words
        # Calculate Shannon entropy
        entropy = -significance_level * np.log2(significance_level) if significance_level > 0 else 0
        # Calculate significance / Shannon entropy
        significance_entropy[word] = entropy / significance_level
    return significance_entropy

# Calculate Shannon entropy score for each article
def sig_entropy(row, column):
    # Get the article topic (assume column name is 'topic')
    article = str(row["topic"])
    # Get the partial text to process (assume column is specified by "column")
    partial = str(row[column])
    
    # Word segmentation
    partial_words = tokenize_chinese(partial.lower())
    words = tokenize_chinese(article.lower())
    
    # Calculate word frequency
    word_duplicates = Counter(words)
    words_sum = sum(word_duplicates.values())
    
    # Compute significance and Shannon entropy
    sig_entropy = compute_significance_and_entropy(partial_words, word_duplicates, words_sum)
    # Calculate the final Shannon entropy score
    score = sum(sig_entropy.values())
    return score

# Read CSV data
Fcsv = pd.read_csv("../dataset/Weibo/val.csv")

print(Fcsv.head())

# Calculate Shannon entropy score for each row and add to new column "sig_entropy_score"
Fcsv["sig_entropy_score"] = Fcsv.apply(lambda x: sig_entropy(x, "topic"), axis=1)

# Calculate average Shannon entropy score
average_shannonKW = Fcsv["sig_entropy_score"].sum() / len(Fcsv)
print(f"Average Shannon entropy score: {average_shannonKW}")

# If you need to save results to a new CSV file
# Fcsv.to_csv(f"~/Desktop/DATALab/minimum/recovery/train_.csv", index=False)
