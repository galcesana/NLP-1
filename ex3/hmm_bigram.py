from collections import defaultdict, Counter
import numpy as np

def train_hmm(train_set):
    # Count tag transitions and word emissions
    transition_counts = defaultdict(Counter)
    emission_counts = defaultdict(Counter)
    tag_counts = Counter()

    for sentence in train_set:
        prev_tag = "<START>"  # Add a start tag for sentence beginning
        for word, tag in sentence:
            word = word.lower()
            tag_counts[tag] += 1
            transition_counts[prev_tag][tag] += 1
            emission_counts[tag][word] += 1
            prev_tag = tag
        transition_counts[prev_tag]["<END>"] += 1  # Add end tag

    # Compute probabilities
    transition_probs = {
        prev_tag: {tag: count / sum(tags.values())
                   for tag, count in tags.items()}
        for prev_tag, tags in transition_counts.items()
    }

    emission_probs = {
        tag: {word: count / sum(words.values())
              for word, count in words.items()}
        for tag, words in emission_counts.items()
    }

    return transition_probs, emission_probs, tag_counts


def viterbi_algorithm(test_set, transition_probs, tag_counts, emission_probs, unknown_tag="NN"):
    tags = list(tag_counts.keys())
    results = []

    for sentence in test_set:
        n = len(sentence)
        sentence_words = [word.lower() for word, _ in sentence]

        # Initialization
        viterbi = np.zeros((len(tags), n))
        backpointer = np.zeros((len(tags), n), dtype=int)

        word = sentence_words[0]
        for i, tag in enumerate(tags):
            transition_prob = transition_probs.get("<START>", {}).get(tag, 0)
            emission_prob = emission_probs.get(tag, {}).get(word, 1e-6)  # Small value for unknown words
            viterbi[i, 0] = transition_prob * emission_prob

        # Recursion
        for t in range(1, n):
            word = sentence_words[t]
            for j, tag in enumerate(tags):
                max_prob, max_state = max(
                    (viterbi[i, t-1] * transition_probs.get(tags[i], {}).get(tag, 0)
                     * emission_probs.get(tag, {}).get(word, 1e-6), i)
                    for i in range(len(tags))
                )
                viterbi[j, t] = max_prob
                backpointer[j, t] = max_state

        # Termination
        max_prob, best_last_tag = max(
            (viterbi[i, n-1] * transition_probs.get(tags[i], {}).get("<END>", 0), i)
            for i in range(len(tags))
        )

        # Backtrack to find the best sequence
        best_sequence = [tags[best_last_tag]]
        for t in range(n-1, 0, -1):
            best_last_tag = backpointer[best_last_tag, t]
            best_sequence.insert(0, tags[best_last_tag])

        results.append(best_sequence)

    return results



def evaluate_hmm(test_set, predicted_tags):
    known_error, unknown_error, total_error = 0, 0, 0
    known_count, unknown_count, total_count = 0, 0, 0

    # Create a set of words from the test set to track known words
    test_words = set(word.lower() for sentence in test_set for word, _ in sentence)

    for sentence, predicted in zip(test_set, predicted_tags):
        for (word, true_tag), predicted_tag in zip(sentence, predicted):
            word = word.lower()

            # Count totals
            total_count += 1
            if word in test_words:
                known_count += 1
                if predicted_tag != true_tag:
                    known_error += 1
            else:
                unknown_count += 1
                if predicted_tag != true_tag:
                    unknown_error += 1

            # Track errors
            if predicted_tag != true_tag:
                total_error += 1

    # Calculate error rates
    known_error_rate = known_error / known_count if known_count > 0 else 0
    unknown_error_rate = unknown_error / unknown_count if unknown_count > 0 else 0
    total_error_rate = total_error / total_count if total_count > 0 else 0

    return {
        "Known Error Rate": known_error_rate,
        "Unknown Error Rate": unknown_error_rate,
        "Total Error Rate": total_error_rate,
    }
