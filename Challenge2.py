
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


#7. Part III: Transformer encoder embeddings

"""
Transformers generate contextual embeddings.

This means the vector for a word depends on the sentence where it appears.

Example:
- "He deposited money in the bank."
- "The fisherman sat on the bank of the river."

The token 'bank' should have different Transformer embeddings in these sentences.
"""


#7.1 Load pretrained Transformer encoder

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer_model = AutoModel.from_pretrained(model_name)

transformer_model.eval()

print(f"\nLoaded Transformer model: {model_name}")


#7.2 Function for contextual word embedding

def get_contextual_word_embedding(sentence, target_word):
    """
    Returns the contextual embedding of a target word inside a sentence.

    This function:
    1. Tokenizes the sentence using a Transformer tokenizer.
    2. Finds tokens matching the target word.
    3. Extracts hidden states from the Transformer.
    4. If the word is split into multiple subword tokens, it averages them.

    Note:
    This simple implementation works best for common words that are not heavily split.
    """

    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = transformer_model(**inputs)

    hidden_states = outputs.last_hidden_state[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    target_tokens = tokenizer.tokenize(target_word)

    matching_indices = []

    for i in range(len(tokens) - len(target_tokens) + 1):
        if tokens[i:i+len(target_tokens)] == target_tokens:
            matching_indices.extend(range(i, i+len(target_tokens)))
            break

    if len(matching_indices) == 0:
        raise ValueError(f"Target word '{target_word}' not found in tokenized sentence: {tokens}")

    embedding = hidden_states[matching_indices].mean(dim=0).numpy()

    return embedding, tokens


#7.3 Contextual example with ambiguous word "bank"

bank_sentences = [
    "He deposited money in the bank.",
    "The bank approved the loan.",
    "The fisherman sat on the bank of the river.",
    "The river bank was covered with grass."
]

bank_embeddings = []

print("\nTransformer tokenization examples:")

for sentence in bank_sentences:
    emb, tokens = get_contextual_word_embedding(sentence, "bank")
    bank_embeddings.append(emb)

    print("\nSentence:", sentence)
    print("Tokens:", tokens)


#7.4 Similarity between contextual bank embeddings

bank_context_results = []

for i in range(len(bank_sentences)):
    for j in range(i + 1, len(bank_sentences)):
        sim = cosine_sim(bank_embeddings[i], bank_embeddings[j])
        bank_context_results.append([
            bank_sentences[i],
            bank_sentences[j],
            sim
        ])

bank_context_df = pd.DataFrame(
    bank_context_results,
    columns=["sentence 1", "sentence 2", "Transformer cosine similarity for 'bank'"]
)

print("\nTransformer contextual similarity for the word 'bank':")
display(bank_context_df.round(3))


#7.5 Compare with Word2Vec

print("""
Comparison:

Word2Vec:
- The word 'bank' always has the same vector.
- Similarity between 'bank' and 'bank' is always 1.0.

Transformer:
- The word 'bank' receives a different vector depending on the sentence.
- Financial-bank contexts should be closer to each other.
- River-bank contexts should be closer to each other.
""")


# ------------------------------------------------------------
# 7.6 Plot contextual Transformer embeddings for "bank"
# ------------------------------------------------------------

bank_labels = [
    "bank: money deposit",
    "bank: loan",
    "bank: fisherman river",
    "bank: river grass"
]

plot_embeddings_2d(
    bank_embeddings,
    bank_labels,
    "Transformer contextual embeddings of the word 'bank'"
)


#8. More examples: same word, different context

"""
Another ambiguous word: apple

Apple can mean:
1. fruit
2. technology company

Word2Vec would store one vector for 'apple'.
Transformer can create different vectors depending on context.
"""

apple_sentences = [
    "I ate a fresh apple for breakfast.",
    "The apple was sweet and juicy.",
    "Apple released a new iPhone model.",
    "Apple is one of the largest technology companies."
]

apple_embeddings = []

for sentence in apple_sentences:
    emb, tokens = get_contextual_word_embedding(sentence, "apple")
    apple_embeddings.append(emb)

apple_context_results = []

for i in range(len(apple_sentences)):
    for j in range(i + 1, len(apple_sentences)):
        sim = cosine_sim(apple_embeddings[i], apple_embeddings[j])
        apple_context_results.append([
            apple_sentences[i],
            apple_sentences[j],
            sim
        ])

apple_context_df = pd.DataFrame(
    apple_context_results,
    columns=["sentence 1", "sentence 2", "Transformer cosine similarity for 'apple'"]
)

print("\nTransformer contextual similarity for the word 'apple':")
display(apple_context_df.round(3))

apple_labels = [
    "apple: fruit breakfast",
    "apple: sweet fruit",
    "apple: iPhone",
    "apple: tech company"
]

plot_embeddings_2d(
    apple_embeddings,
    apple_labels,
    "Transformer contextual embeddings of the word 'apple'"
)


#9. Sentence-level Transformer embeddings

"""
A Transformer can also represent whole sentences.
A simple method is to average token embeddings.

This is not always the best method, but it is enough for demonstration.
"""


def get_sentence_embedding(sentence):
    """
    Returns a sentence embedding by averaging token embeddings.
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = transformer_model(**inputs)

    hidden_states = outputs.last_hidden_state[0]

    attention_mask = inputs["attention_mask"][0].numpy()

    valid_token_embeddings = hidden_states[attention_mask == 1]

    sentence_embedding = valid_token_embeddings.mean(dim=0).numpy()

    return sentence_embedding


sentence_examples = [
    "The doctor treated the patient.",
    "The nurse helped the patient.",
    "A physician cared for a sick person.",
    "The dog chased the cat.",
    "The cat climbed the tree.",
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "I deposited money in the bank.",
    "The fisherman sat on the river bank."
]

sentence_embeddings = [get_sentence_embedding(sentence) for sentence in sentence_examples]

sentence_sim_matrix = cosine_similarity(sentence_embeddings)

sentence_sim_df = pd.DataFrame(
    sentence_sim_matrix,
    columns=sentence_examples,
    index=sentence_examples
)

print("\nTransformer sentence embedding similarity:")
display(sentence_sim_df.round(3))

plot_embeddings_2d(
    sentence_embeddings,
    sentence_examples,
    "Transformer sentence embeddings reduced to 2D using PCA"
)


#10. Direct comparison:
#k-mer vs Word2Vec vs Transformer

comparison_table = pd.DataFrame({
    "Method": [
        "k-mer vectorization",
        "Word2Vec",
        "Transformer encoder"
    ],
    "Representation level": [
        "Fixed fragments / n-grams",
        "Word-level embeddings",
        "Contextual token or sentence embeddings"
    ],
    "Learns meaning?": [
        "No, mostly surface pattern similarity",
        "Partially, from word co-occurrence",
        "Yes, strongly context-dependent"
    ],
    "Context-sensitive?": [
        "No",
        "No",
        "Yes"
    ],
    "Example limitation": [
        "'king' and 'queen' may look unrelated if their character fragments differ",
        "'bank' has one vector for both money bank and river bank",
        "Can still fail on rare words, long reasoning, or domain-specific language"
    ]
})

print("\nSummary comparison table:")
display(comparison_table)


#11. Negative examples / model failure cases

"""
Negative examples are useful in the report.

They show that embeddings are not perfect.
"""


#11.1 Word2Vec negative example

print("\nWord2Vec negative example:")

try:
    result = word2vec_model.wv.most_similar(
        positive=["paris", "poland"],
        negative=["france"],
        topn=5
    )

    print("Expected approximate answer: Warsaw")
    print("Actual results:")

    for word, score in result:
        print(f"{word:12s} {score:.3f}")

except KeyError as e:
    print("Word missing from vocabulary:", e)


print("""
Why this may fail:

The local Word2Vec model was trained on a tiny corpus.
Real Word2Vec models need very large datasets to learn reliable analogies.
Therefore, wrong results are expected and can be discussed as a limitation.
""")


#11.2 Transformer negative example

negative_sentences = [
    "The bat flew out of the cave.",
    "He swung the bat during the baseball game.",
    "The bat is an animal.",
    "The bat was made of wood."
]

bat_embeddings = []

for sentence in negative_sentences:
    emb, tokens = get_contextual_word_embedding(sentence, "bat")
    bat_embeddings.append(emb)

bat_sim_matrix = cosine_similarity(bat_embeddings)

bat_sim_df = pd.DataFrame(
    bat_sim_matrix,
    columns=negative_sentences,
    index=negative_sentences
)

print("\nTransformer contextual similarity for 'bat':")
display(bat_sim_df.round(3))

plot_embeddings_2d(
    bat_embeddings,
    [
        "bat: animal cave",
        "bat: baseball",
        "bat: animal",
        "bat: wooden object"
    ],
    "Transformer contextual embeddings of the word 'bat'"
)

print("""
Transformer limitation:

The Transformer usually captures context better than Word2Vec.
However, it can still produce unexpected similarities.

Reasons:
- The model may rely on broad sentence patterns.
- Simple average or token extraction may not perfectly represent meaning.
- Some contexts may be too short or ambiguous.
""")


#12. Optional: compare Word2Vec and Transformer on the same word

"""
This section directly demonstrates the main difference:

Word2Vec:
bank = one static vector

Transformer:
bank in financial sentence != bank in river sentence
"""

direct_comparison = pd.DataFrame({
    "Example": [
        "Word2Vec: bank in financial sentence",
        "Word2Vec: bank in river sentence",
        "Transformer: bank in financial sentence",
        "Transformer: bank in river sentence"
    ],
    "Vector behavior": [
        "Same vector",
        "Same vector",
        "Contextual vector",
        "Contextual vector"
    ],
    "Can distinguish meaning?": [
        "No",
        "No",
        "Yes",
        "Yes"
    ]
})

print("\nDirect comparison:")
display(direct_comparison)


w2v_bank_similarity = cosine_sim(word2vec_model.wv["bank"], word2vec_model.wv["bank"])
transformer_bank_similarity = cosine_sim(bank_embeddings[0], bank_embeddings[2])

print("\nSimilarity comparison:")
print(f"Word2Vec bank vs bank similarity: {w2v_bank_similarity:.3f}")
print(f"Transformer financial bank vs river bank similarity: {transformer_bank_similarity:.3f}")


#15. Save selected results to CSV files

dna_kmer_df.to_csv("dna_kmer_vectors.csv")
char_kmer_df.to_csv("text_char_kmer_vectors.csv")
kmer_sim_df.to_csv("kmer_similarity.csv")
w2v_df.to_csv("word2vec_similarity_examples.csv")
bank_context_df.to_csv("transformer_bank_context_similarity.csv")
apple_context_df.to_csv("transformer_apple_context_similarity.csv")
comparison_table.to_csv("method_comparison_table.csv", index=False)

print("\nCSV result files saved.")
