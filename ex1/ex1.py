import spacy
from datasets import load_dataset
from collections import Counter, defaultdict
import math


# Preprocessing function
def preprocess(text):
    """Preprocesses a line of text into a list of lemmatized tokens."""
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.is_alpha]


# Unigram and Bigram probability calculations
def unigram_prob(word):
    """Returns the unigram probability (log-space) of a word."""
    count = unigram_counts[word]
    if(count==0):
        return -math.inf
    return math.log(count / total_unigrams)


def bigram_prob(prev_word, word):
    """Returns the bigram probability (log-space) of a word given the previous word."""
    count = bigram_counts[prev_word][word]
    prev_count = unigram_counts[prev_word]
    if(count==0):
        return -math.inf
    return math.log(count / prev_count)


# Interpolated probability calculation using linear interpolation
def interpolate_prob(prev_word, word, alpha_bigram=2/3, alpha_unigram=1/3):
    """Interpolates between unigram and bigram probabilities (log-space)."""
    uni_prob = unigram_prob(word)
    bi_prob = bigram_prob(prev_word, word)
    # Linear interpolation: alpha_bigram * bigram_prob + alpha_unigram * unigram_prob
    try:
        return math.log(alpha_bigram * math.exp(bi_prob) + alpha_unigram * math.exp(uni_prob))
    except: 
        #it means that we tried to take the log of a -inf
        return -math.inf


def compute_sentence_probability(sentence, alpha_bigram=2/3, alpha_unigram=1/3):
    """Computes the log-probability of a sentence using interpolation."""
    words = preprocess(sentence)
    total_log_prob = 0.0
    prev_word = "START"
    for word in words:
        total_log_prob += interpolate_prob(prev_word, word, alpha_bigram, alpha_unigram)
        prev_word = word
    return total_log_prob


# Perplexity calculation
def compute_perplexity(sentences, alpha_bigram=2/3, alpha_unigram=1/3):
    """Computes the perplexity for a list of sentences."""
    total_log_prob = 0.0
    total_words = 0
    for sentence in sentences:
        words = preprocess(sentence)
        total_log_prob += compute_sentence_probability(sentence, alpha_bigram, alpha_unigram)
        total_words += len(words)
    
    return math.exp(-total_log_prob / total_words)


if __name__ == "__main__":
    # Load SpaCy model and Wikitext data
    nlp = spacy.load("en_core_web_sm")
    text_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")

    # Preprocessing the data
    documents = [["START"] + preprocess(line['text']) for line in text_data if 'text' in line]

    # Building unigram and bigram counts
    unigram_counts = Counter()
    bigram_counts = defaultdict(Counter)

    for doc in documents:
        unigram_counts.update(doc)
        for i in range(1, len(doc)):
            bigram_counts[doc[i - 1]][doc[i]] += 1

    # Total unigram count
    total_unigrams = sum(unigram_counts.values())

    # Vocabulary size for smoothing
    vocabulary_size = len(unigram_counts)

    # **Task 2: Continue Sentence using Bigram Model**
    sentence_start = "I have a house in"
    words = preprocess(sentence_start)
    prev_word = words[-1]  # last word in the sentence
    next_word = max(bigram_counts[prev_word], key=lambda w: bigram_counts[prev_word][w])
    print(f"The most probable word following '{sentence_start}' is: '{next_word}'")

    # **Task 3: Sentence probabilities (Bigram Model)**
    sentence_1 = "Brad Pitt was born in Oklahoma"
    sentence_2 = "The actor was born in USA"

    print("Sentence 1 (Bigram Model) probability (log):", compute_sentence_probability(sentence_1,alpha_bigram=1,alpha_unigram=0))
    print("Sentence 2 (Bigram Model) probability (log):", compute_sentence_probability(sentence_2,alpha_bigram=1,alpha_unigram=0))

    # **Task 3b: Perplexity of both sentences combined (using the Bigram model)**
    sentences = [sentence_1, sentence_2]
    perplexity = compute_perplexity(sentences,alpha_bigram=1,alpha_unigram=0)
    print("Perplexity for both sentences (Bigram Model):", perplexity)

    # **Task 4: Sentence probabilities (Interpolated Model)**
    print("Sentence 1 (Interpolated Model) probability (log):", compute_sentence_probability(sentence_1, alpha_bigram=2/3, alpha_unigram=1/3))
    print("Sentence 2 (Interpolated Model) probability (log):", compute_sentence_probability(sentence_2, alpha_bigram=2/3, alpha_unigram=1/3))

    # **Task 4b: Perplexity of both sentences combined (using the Interpolated model)**
    perplexity_interpolated = compute_perplexity(sentences, alpha_bigram=2/3, alpha_unigram=1/3)
    print("Perplexity for both sentences (Interpolated Model):", perplexity_interpolated)
