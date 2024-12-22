from collections import defaultdict, Counter

# def train_baseline1(train_set):
#     tag_count = defaultdict(Counter)
#     for sentence in train_set:
#         for word, tag in sentence:
#             word = word.lower()  # Lowercase for consistency
#             if '+' in tag or '-' in tag:  # Simplify complex tags
#                 tag = tag.split('+')[0].split('-')[0]
#             tag_count[word][tag] += 1
#
#     # Most likely tag for each word
#     most_likely_tags = {word: max(tags, key=tags.get) for word, tags in tag_count.items()}
#
#     return most_likely_tags
#
#
#
#
# def train_baseline2(train_set):
#     # Create a dictionary to store the most likely tag for each word
#     word_to_tag = defaultdict(lambda: 'NN')  # Default tag for unknown words is "NN"
#     # Count occurrences of (word, tag) pairs in the training set
#     word_tag_counts = Counter((word.lower(), tag) for sentence in train_set for word, tag in sentence)
#     # Compute the most likely tag for each word
#     tagged_word_counts = defaultdict(Counter)
#
#     for (word, tag), count in word_tag_counts.items():
#         tagged_word_counts[word][tag] += count
#
#     for word, tag_counts in tagged_word_counts.items():
#         word_to_tag[word] = tag_counts.most_common(1)[0][0]
#     return word_to_tag

def train_baseline(train_set, default_tag='NN', simplify_tags=True):
    """
    Train a baseline model to compute the most likely tag for each word.

    Parameters:
    - train_set: List of tagged sentences (list of (word, tag) tuples).
    - default_tag: Default tag for unknown words (default is 'NN').
    - simplify_tags: Whether to simplify tags by removing parts after '+' or '-' (default is True).

    Returns:
    - word_to_tag: A defaultdict mapping words to their most likely tag. Default value is default_tag.
    """
    # Create a dictionary to count tags for each word
    tag_count = defaultdict(Counter)

    # Count occurrences of each (word, tag) pair in the training set
    for sentence in train_set:
        for word, tag in sentence:
            word = word.lower()  # Convert words to lowercase for consistency
            if simplify_tags:
                # Simplify complex tags by splitting on '+' or '-'
                tag = tag.split('+')[0].split('-')[0]
            tag_count[word][tag] += 1

    # Compute the most likely tag for each word
    word_to_tag = defaultdict(lambda: default_tag)  # Default tag for unknown words
    for word, tags in tag_count.items():
        word_to_tag[word] = max(tags, key=tags.get)  # Select the most frequent tag

    return word_to_tag

def evaluate_baseline(test_set, most_likely_tags, unknown_tag="NN", simplify_tags=True):
    """
    Evaluate a baseline POS tagger on a test set.

    Parameters:
    - test_set: List of tagged sentences (list of (word, tag) tuples).
    - most_likely_tags: Dictionary or defaultdict mapping words to their most likely tag.
    - unknown_tag: Default tag for unknown words (default is 'NN').
    - simplify_tags: Whether to simplify tags by removing parts after '+' or '-' (default is True).

    Returns:
    - A dictionary with error rates:
        - "Known Error Rate": Error rate for known words.
        - "Unknown Error Rate": Error rate for unknown words.
        - "Total Error Rate": Overall error rate.
    """
    # Initialize counters
    known_error, unknown_error, total_error = 0, 0, 0
    known_count, unknown_count, total_count = 0, 0, 0

    for sentence in test_set:
        for word, true_tag in sentence:
            word = word.lower()  # Normalize word case
            if simplify_tags:
                # Simplify tags to remove modifiers
                true_tag = true_tag.split('+')[0].split('-')[0]

            # Get predicted tag, defaulting to `unknown_tag` for unknown words
            predicted_tag = most_likely_tags.get(word, unknown_tag)

            # Check if word is known or unknown
            if word in most_likely_tags:
                known_count += 1
                if predicted_tag != true_tag:
                    known_error += 1
            else:
                unknown_count += 1
                if predicted_tag != true_tag:
                    unknown_error += 1

            # Update total counters
            total_count += 1
            if predicted_tag != true_tag:
                total_error += 1

    # Safeguard against division by zero
    known_error_rate = known_error / known_count if known_count > 0 else 0
    unknown_error_rate = unknown_error / unknown_count if unknown_count > 0 else 0
    total_error_rate = total_error / total_count if total_count > 0 else 0

    return {
        "Known Error Rate": known_error_rate,
        "Unknown Error Rate": unknown_error_rate,
        "Total Error Rate": total_error_rate,
    }
