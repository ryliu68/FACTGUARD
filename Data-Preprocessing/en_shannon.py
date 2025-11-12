import numpy as np
from collections import Counter
import re
# Please convert the json file to csv file format first
def compute_significance_and_entropy(ner_words, word_counts, total_words):
    significance_entropy = {}
    for word in ner_words:
        # (TF)
        word_freq = word_counts[word] if word in word_counts else 1
        significance_level = word_freq / total_words
        # Shannon entropy
        entropy = -significance_level * np.log2(significance_level) if significance_level > 0 else 0
        # significance level / entropy 
        significance_entropy[word] = entropy/significance_level
    return significance_entropy

# Assumes csv has full article in "body_text" column and title in "title" column.
def sig_entropy(row, column):
    # article = str(row["title"]) + " " + str(row["body_text"])
    # article = str(row["label"]) + " " + str(row["body_text"])
    article=str(row["topic"])
    partial = str(row[column])
    partialwords = re.findall(r"\b\w+\b|'\w+", partial.lower())
    words = re.findall(r"\b\w+\b|'\w+", article.lower())
    word_duplicates = Counter(words)
    words_sum = sum(word_duplicates.values())

    sig_entropy = compute_significance_and_entropy(partialwords, word_duplicates, words_sum)
    score = sum(sig_entropy.values())
    return score

import pandas as pd

# Replace with desired CSV
Fcsv = pd.read_csv("../dataset/Weibo21/val.csv")

print(Fcsv.head())

# Replace "keywords" with desired column
Fcsv["sig_entropy_score"] = Fcsv.apply(lambda x: sig_entropy(x, "topic"), axis=1)

#calculate average shannon score
average_shannonKW = Fcsv["sig_entropy_score"].sum() / len(Fcsv)
print(f"Keywords: {average_shannonKW}")

# Fcsv.to_csv(f"~/Desktop/DATALab/minimum/recovery/train_.csv", index=False)
