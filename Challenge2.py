
#1.Please before running install required libraries
# !pip install numpy pandas matplotlib scikit-learn gensim transformers torch


#2.Imports
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from IPython.display import display
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from gensim.models import Word2Vec

import torch
from transformers import AutoTokenizer, AutoModel


#3.Helper functions
def clean_text(text):
    """
    Simple text preprocessing.
    Converts text to lowercase and removes punctuation.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


def tokenize(text):
    """
    Splits cleaned text into tokens.
    """
    return clean_text(text).split()


def cosine_sim(vec1, vec2):
    """
    Calculates cosine similarity between two vectors.
    """
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]


def plot_embeddings_2d(vectors, labels, title):
    """
    Reduces vectors to 2D using PCA and plots them.
    """
    vectors = np.array(vectors)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    plt.figure(figsize=(9, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1])

    for i, label in enumerate(labels):
        plt.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=10)

    plt.title(title)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.grid(True)
    plt.show()


#4.Example corpus

sentences = [
    "The king ruled the kingdom with wisdom",
    "The queen ruled the kingdom with kindness",
    "The prince is the son of the king",
    "The princess is the daughter of the queen",
    "A man can be a king",
    "A woman can be a queen",
    "Paris is the capital of France",
    "Berlin is the capital of Germany",
    "Warsaw is the capital of Poland",
    "France and Germany are countries in Europe",
    "Poland is a country in Europe",
    "The dog chased the cat",
    "The cat climbed the tree",
    "The animal ran through the forest",
    "The bank approved the loan",
    "The fisherman sat on the bank of the river",
    "He deposited money in the bank",
    "The river bank was covered with grass",
    "The apple is a sweet fruit",
    "The orange is a citrus fruit",
    "The doctor treated the patient",
    "The nurse helped the patient",
    "The hospital has doctors and nurses",
]

tokenized_sentences = [tokenize(sentence) for sentence in sentences]

print("Example tokenized sentences:")
for s in tokenized_sentences[:5]:
    print(s)