from transformers import RobertaTokenizer, DistilBertForSequenceClassification, AdamW, DistilBertTokenizer, DistilBertForSequenceClassification, RobertaForSequenceClassification
from data_loader import *

def train_transformer():
    """
    Code for training and evaluating a Transformer model using DistilRoBERTa.
    """
    # Load the tokenizer and model from HuggingFace Transformers
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=2)

    # Initialize the DataManager with full sentences as strings
    data_manager = DataManager(
        data_type=W2V_AVERAGE,  # Not using custom embeddings
        use_sub_phrases=False,  # Use only full sentences
        dataset_path="/content/drive/MyDrive/Colab Notebooks/stanfordSentimentTreebank",
        batch_size=16
    )

    def encode_sentences(sentences):
        """Helper function to encode sentences using the tokenizer."""
        return tokenizer(
            [" ".join(sent.text) for sent in sentences],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

    # Encode the training, validation, and test datasets
    train_sentences = data_manager.sentences[TRAIN]
    val_sentences = data_manager.sentences[VAL]
    test_sentences = data_manager.sentences[TEST]

    train_encodings = encode_sentences(train_sentences)
    val_encodings = encode_sentences(val_sentences)
    test_encodings = encode_sentences(test_sentences)

    train_labels = torch.tensor([sent.sentiment_class for sent in train_sentences]).long()
    val_labels = torch.tensor([sent.sentiment_class for sent in val_sentences]).long()
    test_labels = torch.tensor([sent.sentiment_class for sent in test_sentences]).long()

    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0)

    # Training loop
    device = get_available_device()
    model.to(device)

    n_epochs = 2
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / total_samples
        train_acc = correct_predictions / total_samples

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item() * labels.size(0)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{n_epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    # Test evaluation
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            test_loss += outputs.loss.item() * labels.size(0)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_loss /= test_total
    test_acc = test_correct / test_total

    print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
    return model, data_manager, tokenizer, 128, device

model, data_manager, tokenizer, max_len, device  = train_transformer()



import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

def evaluate_special_subsets_transformer(model, data_manager, tokenizer, device, max_length=128):
    """
    Evaluate the given Transformer model (e.g., DistilRoBERTa) on special subsets
    like negated polarity examples and rare word examples.

    :param model: Trained Transformer model.
    :param data_manager: DataManager object to fetch datasets (and anything else you need).
    :param tokenizer: The Hugging Face tokenizer used for this model.
    :param device: The device on which we run the evaluation (CPU or GPU).
    :param max_length: The maximum sequence length for tokenization.
    """
    print("Evaluating special subsets...")

    # Access the SentimentTreeBank instance from your data manager
    sentiment_dataset = data_manager.sentiment_dataset

    # Retrieve the test set sentences (list of Sentence objects)
    test_sentences = sentiment_dataset.get_test_set()

    # We'll define small helpers to fetch the subset indices as you already have:
    negated_indices = get_negated_polarity_examples(test_sentences)
    rare_indices = get_rare_words_examples(test_sentences, sentiment_dataset)

    # -- Helper function to build a DataLoader for a subset of sentences --
    def create_subset_loader(indices, batch_size=32):
        subset_sentences = [test_sentences[i] for i in indices]
        texts = [" ".join(s.text) for s in subset_sentences]
        labels = [s.sentiment_class for s in subset_sentences]

        # Tokenize all sentences in one go
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        # Create a TensorDataset of (input_ids, attention_mask, labels)
        dataset = TensorDataset(
            inputs["input_ids"],
            inputs["attention_mask"],
            torch.tensor(labels, dtype=torch.long)
        )

        return DataLoader(dataset, batch_size=batch_size)

    # Build DataLoaders for each subset
    negated_loader = create_subset_loader(negated_indices)
    rare_loader = create_subset_loader(rare_indices)

    # Evaluate each subset
    negated_loss, negated_accuracy = evaluate_transformer_subset(model, negated_loader, device)
    print(f"Negated Polarity Subset - Loss: {negated_loss:.4f}, Accuracy: {negated_accuracy:.4f}")

    rare_loss, rare_accuracy = evaluate_transformer_subset(model, rare_loader, device)
    print(f"Rare Words Subset - Loss: {rare_loss:.4f}, Accuracy: {rare_accuracy:.4f}")

    return {
        "negated": {"loss": negated_loss, "accuracy": negated_accuracy},
        "rare_words": {"loss": rare_loss, "accuracy": rare_accuracy},
    }

def evaluate_transformer_subset(model, data_loader, device):
    """
    Evaluate the given model on a provided DataLoader containing (input_ids, attention_mask, labels).

    :param model: The Transformer model.
    :param data_loader: A DataLoader that yields (input_ids, attention_mask, labels).
    :param device: The device (CPU or GPU).
    :return: (average_loss, accuracy) on this subset.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # No gradient needed during evaluation
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass; the model returns a dict-like object with .loss and .logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate loss
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


evaluate_special_subsets_transformer(model, data_manager, tokenizer, device, max_length=128)