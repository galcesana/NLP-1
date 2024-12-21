import nltk
from nltk.corpus import brown
import MLE
import bigram

# Download the Brown corpus
nltk.download('brown')

# Load the "news" category and get tagged sentences
tagged_sentences = brown.tagged_sents(categories='news')

# Split into training (90%) and test (10%) sets
split_index = int(0.9 * len(tagged_sentences))
train_set = tagged_sentences[:split_index]
test_set = tagged_sentences[split_index:]


if __name__ == "__main__":
    # (b) Most Likely Tag Baseline
    most_likely_tags = MLE.train_baseline(train_set)
    baseline_results = MLE.evaluate_baseline(test_set, most_likely_tags)
    print("Baseline Results:")
    for key in baseline_results.keys():
        print(key," = ", baseline_results[key])

    # (c) bigram HMM tagger
    transition_probs, emission_probs = bigram.train_bigram(train_set)
    hmm_results = bigram.viterbi(test_set, transition_probs, emission_probs)
    hmm_error_rate = MLE.evaluate_baseline(hmm_results, most_likely_tags)  # Reuse evaluation method
    print("HMM Tagger Results:")
    for key in hmm_error_rate.keys():
        print(key," = ", hmm_error_rate[key])

    # (d) Add-one smoothing
    transition_probs, emission_probs = bigram.train_bigram(train_set, smoothing=True)
    hmm_results = bigram.viterbi(test_set, transition_probs, emission_probs)
    hmm_error_rate = MLE.evaluate_baseline(hmm_results, most_likely_tags)  # Reuse evaluation method
    print("HMM Tagger add-one smoothing Results:")
    for key in hmm_error_rate.keys():
        print(key," = ", hmm_error_rate[key])