from collections import defaultdict, Counter
import numpy as np
from pseudo_word_mapper import map_to_pseudo_word

def train_bigram(train_set, smoothing=False):
    # Transition probabilities
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)

    for sentence in train_set:
        prev_tag = "<START>"
        for word, tag in sentence:
            word = word.lower()
            if '+' in tag or '-' in tag:
                tag = tag.split('+')[0].split('-')[0]
            transitions[prev_tag][tag] += 1
            emissions[tag][word] += 1
            prev_tag = tag
        transitions[prev_tag]["<END>"] += 1

    # Normalize probabilities with or without smoothing
    if not smoothing:
        # Without smoothing
        transition_probs = {prev_tag: {tag: count / sum(tags.values()) for tag, count in tags.items()} for
                            prev_tag, tags in transitions.items()}
        emission_probs = {tag: {word: count / sum(words.values()) for word, count in words.items()} for
                          tag, words in emissions.items()}
    else:
        # Vocabulary: all unique words in the training set
        vocab = {word.lower() for sentence in train_set for word, _ in sentence}

        # With Add-One smoothing
        tags = list(emissions.keys())

        # Transition probabilities with smoothing
        transition_probs = {
            prev_tag: {
                tag: (count + 1) / (sum(tags.values()) + len(emissions))
                for tag, count in tags.items()
            }
            for prev_tag, tags in transitions.items()
        }
        # Add unseen tags in the transition probabilities
        for prev_tag in transitions:
            for tag in tags:
                if tag not in transition_probs[prev_tag]:
                    transition_probs[prev_tag][tag] = 1 / (sum(transitions[prev_tag].values()) + len(emissions))
        # Emission probabilities with smoothing
        emission_probs = {
            tag: {
                word: (count + 1) / (sum(words.values()) + len(vocab))
                for word, count in words.items()
            }
            for tag, words in emissions.items()
        }
        # Add unseen words in the emission probabilities
        for tag in emissions:
            for word in vocab:
                if word not in emission_probs[tag]:
                    emission_probs[tag][word] = 1 / (sum(emissions[tag].values()) + len(vocab))

    return transition_probs, emission_probs


# c(ii) Implement the Viterbi algorithm
def viterbi(test_set, transition_probs, emission_probs, unknown_tag="NN"):
    results = []
    tags = list(emission_probs.keys())

    for sentence in test_set:
        words = [word.lower() for word, _ in sentence]
        n = len(words)

        # Initialize DP table
        dp = np.zeros((n, len(tags)))
        bp = np.zeros((n, len(tags)), dtype=int)

        # Initialization
        for j, tag in enumerate(tags):
            dp[0, j] = transition_probs.get("<START>", {}).get(tag, 0) * emission_probs.get(tag, {}).get(words[0], 1e-6)

        # Recursion
        for i in range(1, n):
            for j, tag in enumerate(tags):
                max_prob, max_idx = max(
                    (dp[i - 1, k] * transition_probs.get(tags[k], {}).get(tag, 0) * emission_probs.get(tag, {}).get(words[i], 1e-6), k)
                    for k in range(len(tags))
                )
                dp[i, j] = max_prob
                bp[i, j] = max_idx

        # Termination
        max_prob, max_idx = max((dp[n - 1, j] * transition_probs.get(tags[j], {}).get("<END>", 0), j) for j in range(len(tags)))
        best_path = [max_idx]

        # Backtrack
        for i in range(n - 1, 0, -1):
            best_path.append(bp[i, best_path[-1]])

        best_path.reverse()
        results.append([(words[i], tags[best_path[i]]) for i in range(n)])

    return results

def replace_low_frequency_words(train_set, threshold=5):
    word_freq = Counter(word.lower() for sentence in train_set for word, _ in sentence)
    low_freq_words = {word for word, freq in word_freq.items() if freq < threshold}

    def replace_words(sentence):
        return [(map_to_pseudo_word(word) if word.lower() in low_freq_words else word.lower(), tag) for
                word, tag in sentence]

    return [replace_words(sentence) for sentence in train_set]


def train_bigram_with_pseudo_words(train_set, smoothing=False):
    # Replace low-frequency words with pseudo-words
    train_set = replace_low_frequency_words(train_set)
    return train_bigram(train_set, smoothing=smoothing)

def viterbi_with_pseudo_words(test_set, transition_probs, emission_probs):
    results = []
    tags = list(emission_probs.keys())

    for sentence in test_set:
        words = [map_to_pseudo_word(word.lower()) for word, _ in sentence]
        n = len(words)

        # Initialize DP table
        dp = np.zeros((n, len(tags)))
        bp = np.zeros((n, len(tags)), dtype=int)

        # Initialization
        for j, tag in enumerate(tags):
            dp[0, j] = transition_probs.get("<START>", {}).get(tag, 0) * emission_probs.get(tag, {}).get(words[0], 1e-6)

        # Recursion
        for i in range(1, n):
            for j, tag in enumerate(tags):
                max_prob, max_idx = max(
                    (dp[i - 1, k] * transition_probs.get(tags[k], {}).get(tag, 0) * emission_probs.get(tag, {}).get(words[i], 1e-6), k)
                    for k in range(len(tags))
                )
                dp[i, j] = max_prob
                bp[i, j] = max_idx

        # Termination
        max_prob, max_idx = max((dp[n - 1, j] * transition_probs.get(tags[j], {}).get("<END>", 0), j) for j in range(len(tags)))
        best_path = [max_idx]

        # Backtrack
        for i in range(n - 1, 0, -1):
            best_path.append(bp[i, best_path[-1]])

        best_path.reverse()
        results.append([(sentence[i][0], tags[best_path[i]]) for i in range(n)])

    return results
