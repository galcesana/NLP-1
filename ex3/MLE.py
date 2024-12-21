from collections import defaultdict, Counter

def train_baseline(train_set):
    tag_count = defaultdict(Counter)
    for sentence in train_set:
        for word, tag in sentence:
            word = word.lower()  # Lowercase for consistency
            if '+' in tag or '-' in tag:  # Simplify complex tags
                tag = tag.split('+')[0].split('-')[0]
            tag_count[word][tag] += 1

    # Most likely tag for each word
    most_likely_tags = {word: max(tags, key=tags.get) for word, tags in tag_count.items()}

    return most_likely_tags

# (b)ii Evaluate the baseline
def evaluate_baseline(test_set, most_likely_tags):
    known_error, unknown_error, total_error = 0, 0, 0
    known_count, unknown_count, total_count = 0, 0, 0

    # Assign unknown words the tag "NN"
    unknown_tag = "NN"

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
