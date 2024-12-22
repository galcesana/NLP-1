import nltk
from nltk.corpus import brown
import MLE
import bigram
import hmm_bigram
from pseudo_word_mapper import map_to_pseudo_word


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
    print("\nBaseline Results:")
    for key in baseline_results.keys():
        print(key," = ", baseline_results[key])

    # (c) bigram HMM tagger
    # print("\nHMM Tagger Results 1:")
    # print("Training...")
    # transition_probs, emission_probs, tag_counts = hmm_bigram.train_hmm(train_set)
    # print("Evaluating using viterbi..")
    # hmm_results = hmm_bigram.viterbi_algorithm(test_set, transition_probs, tag_counts, emission_probs)
    # print("Calculating results..")
    # hmm_error_rate = hmm_bigram.evaluate_hmm(hmm_results, most_likely_tags)  # Reuse evaluation method
    # for key in hmm_error_rate.keys():
    #     print(key," = ", hmm_error_rate[key])


    print("\nHMM Tagger Results:")
    print("Training...")
    transition_probs, emission_probs = bigram.train_bigram(train_set)
    print("Evaluating using viterbi..")
    hmm_results = bigram.viterbi(test_set, transition_probs, emission_probs)
    print("Calculating results..")
    hmm_error_rate = hmm_bigram.evaluate_hmm(hmm_results, most_likely_tags)  # Reuse evaluation method
    for key in hmm_error_rate.keys():
        print(key," = ", hmm_error_rate[key])

    # (d) Add-one smoothing
    transition_probs, emission_probs = bigram.train_bigram(train_set, smoothing=True)
    hmm_results = bigram.viterbi(test_set, transition_probs, emission_probs)
    hmm_error_rate = MLE.evaluate_baseline(hmm_results, most_likely_tags)  # Reuse evaluation method
    print("\nHMM Tagger add-one smoothing Results:")
    for key in hmm_error_rate.keys():
        print(key," = ", hmm_error_rate[key])
    #
    # # (e)(ii) Running Viterbi with Pseudo-Words
    # # Replace low-frequency words in the training set with pseudo-words
    # train_set_pseudo = bigram.replace_low_frequency_words(train_set)
    #
    # # Train with pseudo-words and MLE
    # transition_probs, emission_probs = bigram.train_bigram_with_pseudo_words(train_set_pseudo)
    # hmm_results_pseudo = bigram.viterbi_with_pseudo_words(test_set, transition_probs, emission_probs)
    #
    # # Evaluate
    # pseudo_mle_error_rate = MLE.evaluate_baseline(hmm_results_pseudo, most_likely_tags)
    # print("\nHMM Tagger Results with Pseudo-Words (MLE):")
    # for key in pseudo_mle_error_rate.keys():
    #     print(key, " = ", pseudo_mle_error_rate[key])
    #
    # # (e)(iii) Add-One Smoothing with Pseudo-Words
    # # Train with pseudo-words and Add-One smoothing
    # transition_probs, emission_probs = bigram.train_bigram_with_pseudo_words(train_set_pseudo, smoothing=True)
    # hmm_results_pseudo_smoothed = bigram.viterbi_with_pseudo_words(test_set, transition_probs, emission_probs)
    #
    # # Evaluate
    # pseudo_smoothed_error_rate = MLE.evaluate_baseline(hmm_results_pseudo_smoothed, most_likely_tags)
    # print("HMM Tagger Results with Pseudo-Words and Add-One Smoothing:")
    # for key in pseudo_smoothed_error_rate.keys():
    #     print(key, " = ", pseudo_smoothed_error_rate[key])
    #
    #
    # from sklearn.metrics import confusion_matrix, classification_report
    #
    #
    # def get_confusion_matrix(test_set, predictions):
    #     true_tags = []
    #     pred_tags = []
    #
    #     for sentence, pred_sentence in zip(test_set, predictions):
    #         for (_, true_tag), (_, pred_tag) in zip(sentence, pred_sentence):
    #             true_tags.append(true_tag)
    #             pred_tags.append(pred_tag)
    #
    #     tags = list(set(true_tags + pred_tags))
    #     cm = confusion_matrix(true_tags, pred_tags, labels=tags)
    #     report = classification_report(true_tags, pred_tags, labels=tags)
    #     return cm, report
    #
    #
    # cm, report = get_confusion_matrix(test_set, hmm_results_pseudo_smoothed)
    # print("Confusion Matrix:\n", cm)
    # print("\nClassification Report:\n", report)
    #
