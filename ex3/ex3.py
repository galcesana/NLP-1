import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
import numpy as np

# Download the Brown corpus
nltk.download('brown')

# Load the "news" category and get tagged sentences
tagged_sentences = brown.tagged_sents(categories='news')

# Split into training (90%) and test (10%) sets
split_index = int(0.9 * len(tagged_sentences))
train_set = tagged_sentences[:split_index]
test_set = tagged_sentences[split_index:]



# (b)i Compute the most likely tag for each word
tag_count = defaultdict(Counter)
for sentence in train_set:
    for word, tag in sentence:
        word = word.lower()  # Lowercase for consistency
        if '+' in tag or '-' in tag:  # Simplify complex tags
            tag = tag.split('+')[0].split('-')[0]
        tag_count[word][tag] += 1

# Most likely tag for each word
most_likely_tags = {word: max(tags, key=tags.get) for word, tags in tag_count.items()}

# Assign unknown words the tag "NN"
unknown_tag = "NN"

# (b)ii Evaluate the baseline
def evaluate_baseline(test_set, most_likely_tags):
    known_error, unknown_error, total_error = 0, 0, 0
    known_count, unknown_count, total_count = 0, 0, 0

    for sentence in test_set:
        for word, true_tag in sentence:
            word = word.lower()
            if '+' in true_tag or '-' in true_tag:
                true_tag = true_tag.split('+')[0].split('-')[0]

            predicted_tag = most_likely_tags.get(word, unknown_tag)
            if word in most_likely_tags:
                known_count += 1
                if predicted_tag != true_tag:
                    known_error += 1
            else:
                unknown_count += 1
                if predicted_tag != true_tag:
                    unknown_error += 1
            total_count += 1
            if predicted_tag != true_tag:
                total_error += 1

    return {
        "Known Error Rate": known_error / known_count,
        "Unknown Error Rate": unknown_error / unknown_count,
        "Total Error Rate": total_error / total_count,
    }

baseline_results = evaluate_baseline(test_set, most_likely_tags)
print("Baseline Results:", baseline_results)



# (c-i) Bigram HMM Tagger
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

# Normalize to probabilities
transition_probs = {prev_tag: {tag: count / sum(tags.values()) for tag, count in tags.items()} for prev_tag, tags in transitions.items()}
emission_probs = {tag: {word: count / sum(words.values()) for word, count in words.items()} for tag, words in emissions.items()}

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

# c(iii) Evaluate the HMM tagger

hmm_results = viterbi(test_set, transition_probs, emission_probs)
hmm_error_rate = evaluate_baseline(hmm_results, most_likely_tags)  # Reuse evaluation method
print("HMM Tagger Results:", hmm_error_rate)


# (d) Add-One Smoothing
vocab = {word for sentence in train_set for word, _ in sentence}
laplace_emission_probs = {tag: defaultdict(lambda: 1 / (len(vocab) + len(words))) for tag, words in emissions.items()}
laplace_transition_probs = {prev_tag: defaultdict(lambda: 1 / (len(tags) + len(prev_tags))) for prev_tag, prev_tags in transitions.items()}
