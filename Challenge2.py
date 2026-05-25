
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


#6.Part II: Word2Vec embeddings

"""
Word2Vec learns word vectors from word co-occurrence.
Words appearing in similar contexts receive similar vectors.

Important property:
Each word has one fixed vector.

For example:
- 'bank' has one vector
- the vector is the same in 'money bank' and 'river bank'

This is different from Transformers.
"""


#6.1 Train Word2Vec model

word2vec_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=4,
    sg=1,
    epochs=300,
    seed=42
)

print("\nWord2Vec vocabulary:")
print(list(word2vec_model.wv.index_to_key))


#6.2 Show most similar words

query_words = ["king", "queen", "bank", "doctor", "fruit", "capital"]

for word in query_words:
    print(f"\nMost similar words to '{word}':")
    similar_words = word2vec_model.wv.most_similar(word, topn=5)
    for similar_word, score in similar_words:
        print(f"{similar_word:12s} {score:.3f}")


#6.3 Word2Vec similarity examples

word_pairs = [
    ("king", "queen"),
    ("king", "dog"),
    ("doctor", "nurse"),
    ("france", "germany"),
    ("apple", "orange"),
    ("bank", "money"),
    ("bank", "river"),
]

w2v_results = []

for w1, w2 in word_pairs:
    sim = cosine_sim(word2vec_model.wv[w1], word2vec_model.wv[w2])
    w2v_results.append([w1, w2, sim])

w2v_df = pd.DataFrame(w2v_results, columns=["word 1", "word 2", "Word2Vec cosine similarity"])

print("\nWord2Vec similarity examples:")
display(w2v_df.round(3))


#6.4 Plot selected Word2Vec embeddings

selected_words = [
    "king", "queen", "prince", "princess",
    "man", "woman",
    "france", "germany", "poland",
    "paris", "berlin", "warsaw",
    "doctor", "nurse", "patient",
    "dog", "cat",
    "bank", "river", "money"
]

selected_vectors = [word2vec_model.wv[word] for word in selected_words]

plot_embeddings_2d(
    selected_vectors,
    selected_words,
    "Word2Vec embeddings reduced to 2D using PCA"
)


# 6.5 Word2Vec vector arithmetic

"""
Classic example:
king - man + woman ≈ queen

Because this is a tiny corpus, results may not be perfect.
This is useful because the assignment also allows negative examples.
"""

print("\nVector arithmetic: king - man + woman")
result = word2vec_model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=5
)

for word, score in result:
    print(f"{word:12s} {score:.3f}")


print("\nVector arithmetic: paris - france + germany")
result = word2vec_model.wv.most_similar(
    positive=["paris", "germany"],
    negative=["france"],
    topn=5
)

for word, score in result:
    print(f"{word:12s} {score:.3f}")


#6.6 Word2Vec limitation: one vector per word

bank_vector_1 = word2vec_model.wv["bank"]
bank_vector_2 = word2vec_model.wv["bank"]

print("\nWord2Vec vector for 'bank' is always the same.")
print("Cosine similarity between 'bank' in two different contexts:")
print(cosine_sim(bank_vector_1, bank_vector_2))

print("""
Word2Vec limitation:

The word 'bank' has one embedding.
Therefore, Word2Vec cannot directly distinguish:

1. bank = financial institution
2. bank = side of a river

The model can learn that 'bank' is related to both 'money' and 'river',
but it still stores only one mixed representation for the word.
""")