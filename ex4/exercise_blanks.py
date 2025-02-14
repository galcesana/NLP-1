import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import matplotlib.pyplot as plt


# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a must...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    return


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    vec = np.zeros(size, dtype=np.float32)
    vec[ind] = 1.0
    return vec


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    # We'll store the sum of the one-hot vectors here
    vocab_size = len(word_to_ind)
    sum_vec = np.zeros(vocab_size, dtype=np.float32)

    # Count how many valid tokens we have (i.e., tokens that appear in word_to_ind)
    valid_tokens = 0

    for word in sent.text:
        if word in word_to_ind:
            idx = word_to_ind[word]
            # increment the sum_vec at the index
            sum_vec += get_one_hot(vocab_size, idx)
            valid_tokens += 1

    # Compute the average if we had at least one valid token
    if valid_tokens > 0:
        sum_vec /= valid_tokens

    return sum_vec


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {word: idx for idx, word in enumerate(words_list)}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    return


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape




# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        return

    def forward(self, text):
        return

    def predict(self, text):
        return


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        """
        :param embedding_dim: dimension of the average embedding input
        """
        super(LogLinear, self).__init__()
        # A single linear layer that outputs a single score
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        """
        :param x: A tensor of shape [batch_size, embedding_dim]
        :return: A tensor of shape [batch_size], the raw logits
        """
        logits = self.linear(x).squeeze(1)  # squeeze to get shape [batch_size]
        return logits

    def predict(self, x):
        """
        Computes the forward pass, applies a sigmoid, and thresholds at 0.5
        to obtain a binary classification (0 or 1).

        :param x: A tensor of shape [batch_size, embedding_dim]
        :return: A tensor of shape [batch_size], with 0/1 predictions
        """
        logits = self.forward(x)  # raw logits
        probs = torch.sigmoid(logits)  # convert logits to probabilities
        preds = (probs >= 0.5).long()  # threshold at 0.5
        return preds


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    # Convert predictions to binary by thresholding at 0.5
    preds_binary = (preds >= 0.5).astype(int)

    # Count how many predictions match the true labels
    correct = (preds_binary == y).sum()

    # Calculate the accuracy
    accuracy = correct / len(y)
    return accuracy


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """

    # Put the model in training mode
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Iterate over all batches in the training data
    for inputs, labels in data_iterator:
        # Move inputs and labels to the device (GPU/CPU) if necessary
        # inputs, labels = inputs.to(device), labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass: get logits from the model
        logits = model(inputs)

        # Compute the loss
        loss = criterion(logits, labels)

        # Backward pass: compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate loss (multiplied by batch size for correct averaging later)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        # Convert logits to predictions for accuracy
        # For binary classification (BCEWithLogitsLoss):
        # preds = (torch.sigmoid(logits) >= 0.5).long()
        # For multi-class (CrossEntropyLoss with shape [batch_size, num_classes]):
        #   preds = torch.argmax(logits, dim=1)
        preds = (torch.sigmoid(logits) >= 0.5).long()

        # Compare predictions with true labels
        correct = (preds == labels).sum().item()
        total_correct += correct
        total_samples += batch_size

    # Compute average loss and accuracy over the epoch
    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples

    return average_loss, average_accuracy


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    # Set model to evaluation mode
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_iterator:
            # Move data to the appropriate device
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # Forward pass: compute logits
            logits = model(inputs)

            # Compute loss
            loss = criterion(logits, labels)

            # Accumulate total loss
            batch_size = labels.size()[0]
            total_loss += loss.item() * batch_size

            # Convert logits to predictions
            preds = (torch.sigmoid(logits) >= 0.5).long()

            # Calculate number of correct predictions
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += batch_size

    # Compute average loss and accuracy
    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples

    return average_loss, average_accuracy


def get_predictions_for_data(model, data_iter):
    """
    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    model.eval()  # Ensure the model is in evaluation mode
    all_preds = []

    # Disable gradient tracking for evaluation
    with torch.no_grad():
        for inputs, _ in data_iter:
            # Use the model's `predict` method to get discrete predictions
            batch_preds = model.predict(inputs)  # shape: [batch_size]

            # Optionally move predictions to CPU if you're on GPU
            all_preds.append(batch_preds.cpu())

    # Concatenate all predictions into a single 1D tensor
    all_preds = torch.cat(all_preds, dim=0)

    # Return as a PyTorch tensor or convert to numpy
    # return all_preds            # as torch tensor
    return all_preds.numpy()      # as numpy array


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    # 1) Set up the optimizer (Adam) and the loss function (BCEWithLogitsLoss for binary classification)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # 2) Prepare lists to store metrics across epochs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # 3) For each epoch, train on the training set, then evaluate on the validation set
    for epoch in range(n_epochs):
        # --- Training ---
        train_iter = data_manager.get_torch_iterator(TRAIN)
        train_loss, train_acc = train_epoch(model, train_iter, optimizer, criterion)

        # --- Validation ---
        val_iter = data_manager.get_torch_iterator(VAL)
        val_loss, val_acc = evaluate(model, val_iter, criterion)

        # --- Store metrics ---
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Print current epoch metrics
        print(f"[Epoch {epoch+1}/{n_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 4) Return the lists of metrics
    return train_losses, train_accuracies, val_losses, val_accuracies


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    # 1) Instantiate a DataManager set up for one-hot average embeddings
    data_manager = DataManager(
        data_type=ONEHOT_AVERAGE,  # one-hot average
        use_sub_phrases=True,  # include sub-phrases in training
        dataset_path="stanfordSentimentTreebank",  # path to the dataset
        batch_size=64  # batch size
    )

    # 2) Determine the input dimension from data_manager
    input_dim = data_manager.get_input_shape()[0]

    # 3) Create the log-linear model (single linear layer expecting input_dim)
    model = LogLinear(embedding_dim=input_dim)

    # 4) Define hyperparameters for training
    n_epochs = 20
    lr = 0.01
    weight_decay = 0.001


    # 5) Train the model using train_model (which internally calls train_epoch & evaluate)
    train_losses, train_accs, val_losses, val_accs = train_model(
        model=model,
        data_manager=data_manager,
        n_epochs=n_epochs,
        lr=lr,
        weight_decay=weight_decay
    )

    # Return relevant info for plotting
    return model, train_losses, train_accs, val_losses, val_accs

def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    return


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    return


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # -- Plot Loss --
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -- Plot Accuracy --
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title("Accuracy vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    model, train_losses, train_accuracies, val_losses, val_accuracies = train_log_linear_with_one_hot()
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    # train_log_linear_with_w2v()
    # train_lstm_with_w2v()