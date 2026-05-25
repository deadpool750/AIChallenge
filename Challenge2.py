
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


#5. Part I: k-mer based vectorization

"""
In bioinformatics, k-mers are fixed-length subsequences.
For example, for DNA sequence "ATGCG" and k = 3:

ATG, TGC, GCG

In text, a similar idea can be used with character n-grams.
This representation counts local fragments but does not understand meaning.
"""


#5.1DNA k-mer example

dna_sequences = [
    "ATGCGATACG",
    "ATGCGATACC",
    "TTTTGGGCCC",
    "GCGCGCGCAA",
    "ATATATATAT",
]

def get_kmers(sequence, k=3):
    """
    Returns all k-mers from a sequence.
    """
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


k = 3

dna_kmer_counts = []

for seq in dna_sequences:
    kmers = get_kmers(seq, k)
    counts = Counter(kmers)
    dna_kmer_counts.append(counts)

print("\nDNA k-mer counts:")
for seq, counts in zip(dna_sequences, dna_kmer_counts):
    print(seq, counts)


#5.2 Convert DNA k-mers to table

all_kmers = sorted(set(kmer for counts in dna_kmer_counts for kmer in counts))

dna_kmer_matrix = []

for counts in dna_kmer_counts:
    row = [counts.get(kmer, 0) for kmer in all_kmers]
    dna_kmer_matrix.append(row)

dna_kmer_df = pd.DataFrame(dna_kmer_matrix, columns=all_kmers, index=dna_sequences)

print("\nDNA k-mer vectorization table:")
display(dna_kmer_df)


#5.3 Text character k-mer example

text_examples = [
    "king",
    "queen",
    "kingdom",
    "dog",
    "cat",
    "bank",
    "riverbank",
]

char_vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3))
char_kmer_matrix = char_vectorizer.fit_transform(text_examples)

char_kmer_df = pd.DataFrame(
    char_kmer_matrix.toarray(),
    columns=char_vectorizer.get_feature_names_out(),
    index=text_examples
)

print("\nText character 3-mer table:")
display(char_kmer_df)


#5.4 Similarity using k-mer vectors

kmer_sim_matrix = cosine_similarity(char_kmer_matrix)

kmer_sim_df = pd.DataFrame(
    kmer_sim_matrix,
    columns=text_examples,
    index=text_examples
)

print("\nCosine similarity based on character 3-mers:")
display(kmer_sim_df.round(3))


#5.5 Interpretation

print("""
Interpretation of k-mer vectorization:

k-mer based representation is simple and useful when local fragments matter.
For example, DNA sequences with similar short fragments will have similar vectors.

However, k-mer vectorization does not understand semantic meaning.
For text, 'king' and 'queen' are semantically related, but character k-mers
do not necessarily show this relationship well.

This is the key limitation:
k-mer vectors capture surface patterns, not meaning.
""")


