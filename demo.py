import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    EarlyStopping,
    LearningRateMonitor
)
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split, ConcatDataset
import numpy as np
import random
import re
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Constants
INPUT_DIM = 3000  # 10 tokens * 300-dim embeddings
HIDDEN_DIM = 128
OUTPUT_DIM = 1
DROPOUT_PROB = 0.5
NUM_SAMPLES = 2000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
MC_SAMPLES = 100  # Number of Monte Carlo samples for uncertainty

# Load GloVe embeddings
def load_glove_embeddings(glove_file_path='glove.6B.300d.txt'):
    """
    Loads GloVe embeddings from a file.

    Args:
        glove_file_path (str): Path to the GloVe embeddings file.

    Returns:
        dict: A dictionary mapping words to their embedding tensors.
    """
    embeddings = {}
    try:
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
                embeddings[word] = vector
        print("GloVe embeddings successfully loaded.")
    except FileNotFoundError:
        print(f"Error: The GloVe file '{glove_file_path}' was not found.")
        print("Please download it from https://nlp.stanford.edu/projects/glove/ and place it in the current directory.")
        exit(1)
    return embeddings

# Synthetic Text Dataset Generation with Enhanced Diversity
class SentimentDataset(Dataset):
    """
    A synthetic dataset for sentiment analysis with diverse sentences.
    """

    def __init__(self, num_samples=1000, embeddings=None):
        """
        Initializes the dataset with positive and negative sentences.

        Args:
            num_samples (int): Number of samples in the dataset.
            embeddings (dict): A dictionary mapping words to their embedding tensors.
        """
        self.embeddings = embeddings if embeddings else {}
        self.positive_sentences = [
            "I love this product!",
            "Absolutely fantastic experience.",
            "Highly recommend to everyone.",
            "Exceeded my expectations.",
            "Will buy again.",
            "Five stars for sure!",
            "Incredible performance!",
            "I'm very satisfied.",
            "Top-notch service.",
            "Amazing quality and support."
        ]
        self.negative_sentences = [
            "This is terrible.",
            "Absolutely horrible experience.",
            "Do not recommend to anyone.",
            "Failed to meet my expectations.",
            "Will never buy again.",
            "One star is too much.",
            "Incredible disappointment!",
            "I'm very dissatisfied.",
            "Poor service and quality.",
            "Worst product ever."
        ]
        # Generate samples
        self.texts = []
        self.labels = []
        for i in range(num_samples):
            if i % 2 == 0:
                sentence = random.choice(self.positive_sentences)
                label = 1
            else:
                sentence = random.choice(self.negative_sentences)
                label = 0
            self.texts.append(sentence)
            self.labels.append(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves the tokenized and embedded representation of a sentence along with its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (input_vector, label)
        """
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized_text = self.tokenize(text)
        return tokenized_text, torch.tensor(label, dtype=torch.float32)

    def tokenize(self, text):
        """
        Cleans, tokenizes, and maps a sentence to its GloVe embeddings.

        Args:
            text (str): The input sentence.

        Returns:
            torch.Tensor: A flattened tensor of concatenated embeddings.
        """
        # Remove punctuation and lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = word_tokenize(text)
        embeddings_list = []
        for token in tokens:
            if token in self.embeddings:
                embeddings_list.append(self.embeddings[token])
            else:
                print(f"Token '{token}' not found in embeddings. Assigning zero vector.")
                embeddings_list.append(torch.zeros(300))
        # Pad or truncate to fixed length (10 tokens)
        max_tokens = 10
        if len(embeddings_list) < max_tokens:
            embeddings_list += [torch.zeros(300) for _ in range(max_tokens - len(embeddings_list))]
        else:
            embeddings_list = embeddings_list[:max_tokens]
        # Stack into tensor and flatten
        return torch.stack(embeddings_list).view(-1)  # Shape: (3000,)

# Define a LightningDataModule for better data handling
class SentimentDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling sentiment analysis data.
    """

    def __init__(self, batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES, input_dim=INPUT_DIM, embeddings=None):
        """
        Initializes the DataModule with batch size, number of samples, and embeddings.

        Args:
            batch_size (int): Number of samples per batch.
            num_samples (int): Number of samples in the dataset.
            input_dim (int): Dimension of the input vectors.
            embeddings (dict): A dictionary mapping words to their embedding tensors.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.embeddings = embeddings
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        """
        Prepares data by loading embeddings.
        """
        if not self.embeddings:
            self.embeddings = load_glove_embeddings()

    def setup(self, stage=None):
        """
        Sets up the dataset by creating training and validation splits.
        """
        if self.train_dataset is not None and self.val_dataset is not None:
            # Datasets are already set, do not reset them
            return
        dataset = SentimentDataset(num_samples=self.num_samples, embeddings=self.embeddings)
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def collate_fn(self, batch):
        """
        Custom collate function to stack inputs and labels.

        Args:
            batch (list): List of tuples (input_vector, label).

        Returns:
            tuple: (stacked_inputs, stacked_labels)
        """
        x, y = zip(*batch)
        x = torch.stack(x)
        y = torch.stack(y)
        return x, y

    def train_dataloader(self):
        """
        Returns the training DataLoader.

        Returns:
            DataLoader: Training DataLoader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def add_feedback(self, feedback_texts, feedback_labels):
        """
        Incorporates feedback data into the training dataset.

        Args:
            feedback_texts (list): List of feedback text strings.
            feedback_labels (list): List of feedback labels (0 or 1).
        """
        assert len(feedback_texts) == len(feedback_labels), "Feedback texts and labels must be the same length."
        print("\n--- Incorporating Feedback Data ---")
        for i, (text, label) in enumerate(zip(feedback_texts, feedback_labels)):
            print(f"Feedback {i+1}: Text - \"{text}\", Label - {'Positive' if label == 1 else 'Negative'}")
        print("------------------------------------\n")

        # Tokenize feedback data
        tokenized_feedback = []
        for text in feedback_texts:
            # Clean and tokenize
            cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
            tokens = word_tokenize(cleaned_text)
            embeddings_list = []
            for token in tokens:
                if token in self.embeddings:
                    embeddings_list.append(self.embeddings[token])
                else:
                    print(f"Feedback Token '{token}' not found in embeddings. Assigning zero vector.")
                    embeddings_list.append(torch.zeros(300))
            # Pad or truncate
            max_tokens = 10
            if len(embeddings_list) < max_tokens:
                embeddings_list += [torch.zeros(300) for _ in range(max_tokens - len(embeddings_list))]
            else:
                embeddings_list = embeddings_list[:max_tokens]
            # Stack and flatten
            tokenized = torch.stack(embeddings_list).view(-1)
            tokenized_feedback.append(tokenized)

        # Convert to tensors
        x_feedback = torch.stack(tokenized_feedback)
        y_feedback = torch.tensor(feedback_labels, dtype=torch.float32)

        # Create a TensorDataset for feedback
        feedback_tensor_dataset = TensorDataset(x_feedback, y_feedback)

        # Concatenate feedback with existing training data
        self.train_dataset = ConcatDataset([self.train_dataset, feedback_tensor_dataset])
        print(f"Training dataset size after adding feedback: {len(self.train_dataset)} samples.\n")

# Bayesian Neural Network using Monte Carlo Dropout
class BayesianNN(pl.LightningModule):
    """
    A Bayesian Neural Network for binary sentiment classification using Monte Carlo Dropout.
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, p=DROPOUT_PROB, lr=LEARNING_RATE):
        """
        Initializes the Bayesian Neural Network with specified architecture.

        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Number of neurons in hidden layers.
            output_dim (int): Dimension of the output layer.
            p (float): Dropout probability.
            lr (float): Learning rate.
        """
        super(BayesianNN, self).__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        """
        Defines the forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
            batch (tuple): Batch of data (inputs, labels).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        acc = ((preds >= 0.5) == y.byte()).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Args:
            batch (tuple): Batch of data (inputs, labels).
            batch_idx (int): Index of the batch.
        """
        x, y = batch
        logits = self(x).squeeze()
        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)
        acc = ((preds >= 0.5) == y.byte()).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            dict: Dictionary containing optimizer and scheduler.
        """
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=3,
            factor=0.5,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def predict_sentiment(self, x, num_samples=MC_SAMPLES):
        """
        Makes predictions using Monte Carlo Dropout to estimate uncertainty.

        Args:
            x (torch.Tensor): Input tensor.
            num_samples (int): Number of forward passes to perform.

        Returns:
            tuple: (mean_predictions, uncertainties)
        """
        self.train()  # Enable dropout
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self(x)
                preds = torch.sigmoid(logits)
                predictions.append(preds)
        predictions = torch.stack(predictions)  # Shape: (MC_SAMPLES, batch_size, 1)
        mean_prediction = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        self.eval()  # Set back to eval mode
        return mean_prediction, uncertainty

def main():
    """
    Main function to execute the training, prediction, and feedback process.
    """
    # Load GloVe embeddings
    print("Loading GloVe embeddings...")
    embeddings = load_glove_embeddings('glove.6B.300d.txt')  # Ensure this file is in the current directory
    print("GloVe embeddings loaded.\n")

    # Initialize DataModule
    data_module = SentimentDataModule(embeddings=embeddings)
    data_module.setup()

    # Display sample data from the dataset
    print("\n--- Sample Data from SentimentDataset ---")
    # Accessing the original dataset
    original_dataset = data_module.train_dataset.dataset
    for i in range(3):
        sample_text = original_dataset.texts[i]
        sample_label = original_dataset.labels[i]
        print(f"Sample {i+1}:")
        print(f"  Text: {sample_text}")
        print(f"  Label: {'Positive' if sample_label == 1 else 'Negative'}")
    print("------------------------------------------\n")

    # Initialize and train the model
    model = BayesianNN()

    # Define callbacks
    progress_bar = TQDMProgressBar(refresh_rate=20)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=True,  # Enable logger (TensorBoard)
        enable_checkpointing=False,
        callbacks=[progress_bar, early_stop_callback, lr_monitor],
        deterministic=True,  # For reproducibility
        log_every_n_steps=10  # Adjust as needed
    )

    print("Starting initial training...")
    trainer.fit(model, datamodule=data_module)
    print("Initial training completed.\n")

    # Prepare multiple test inputs
    test_texts = [
        "This product is amazing!",
        "I hate this item.",
        "It's okay, not great.",
        "Absolutely fantastic experience.",
        "Very disappointed with the product.",
        "Exceeded my expectations!",
        "Terrible customer service.",
        "I love it!",
        "Worst purchase ever.",
        "Highly recommend to everyone.",
        "Not worth the price.",
        "I'm very satisfied."
    ]

    # Tokenize and pad test inputs
    tokenized_tests = []
    for text in test_texts:
        # Clean and tokenize
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = word_tokenize(cleaned_text)
        embeddings_list = []
        for token in tokens:
            if token in embeddings:
                embeddings_list.append(embeddings[token])
            else:
                print(f"Test Token '{token}' not found in embeddings. Assigning zero vector.")
                embeddings_list.append(torch.zeros(300))
        # Pad or truncate
        max_tokens = 10
        if len(embeddings_list) < max_tokens:
            embeddings_list += [torch.zeros(300) for _ in range(max_tokens - len(embeddings_list))]
        else:
            embeddings_list = embeddings_list[:max_tokens]
        # Stack and flatten
        tokenized = torch.stack(embeddings_list).view(-1)
        tokenized_tests.append(tokenized)
    x_tests = torch.stack(tokenized_tests)

    # Make predictions before feedback
    print("--- Predictions Before Feedback ---")
    model.eval()
    mean_preds, uncertainties = model.predict_sentiment(x_tests)
    for i, text in enumerate(test_texts):
        pred = mean_preds[i].item()
        uncertainty = uncertainties[i].item()
        sentiment = "Positive" if pred >= 0.5 else "Negative"
        print(f"Text: \"{text}\"")
        print(f"  Prediction: {pred:.4f} ({sentiment})")
        print(f"  Uncertainty: {uncertainty:.4f}\n")
    print("------------------------------------\n")

    # Set the model back to train mode
    model.train()

    # Feedback - retrain with additional diverse data (balanced)
    feedback_texts = [
        "This product is amazing!",
        "Absolutely love it!",
        "Fantastic quality and great support.",
        "Exceeded my expectations!",
        "Will buy again.",
        "Highly recommend to everyone.",
        "Five stars for sure!",
        "Incredible performance!",
        "I'm very satisfied.",
        "Top-notch service.",
        "Not worth the price.",
        "Very disappointed with the product.",
        "Terrible customer service.",
        "I hate this item.",
        "Worst purchase ever.",
        "It's okay, could be better.",
        "I feel neutral about this.",
        "Nothing special, just average.",
        "Mediocre performance.",
        "Couldn’t be happier with this."
    ]
    feedback_labels = [1]*10 + [0]*10  # 10 positive and 10 negative

    print("--- Feedback Data ---")
    for i, text in enumerate(feedback_texts):
        print(f"Feedback {i+1}:")
        print(f"  Text: {text}")
        print(f"  Label: {'Positive' if feedback_labels[i] == 1 else 'Negative'}")
    print("---------------------\n")

    # Incorporate feedback data into the DataModule
    data_module.add_feedback(feedback_texts, feedback_labels)

    # Retrain the model with feedback data
    print("Incorporating feedback and retraining the model...")

    # Create a new Trainer instance for retraining
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=True,  # Enable logger (TensorBoard)
        enable_checkpointing=False,
        callbacks=[progress_bar, early_stop_callback, lr_monitor],
        deterministic=True,  # For reproducibility
        log_every_n_steps=10  # Adjust as needed
    )

    trainer.fit(model, datamodule=data_module)
    print("Retraining with feedback completed.\n")

    # Make predictions after feedback
    print("--- Predictions After Feedback ---")
    model.eval()
    mean_preds_after, uncertainties_after = model.predict_sentiment(x_tests)
    for i, text in enumerate(test_texts):
        pred = mean_preds_after[i].item()
        uncertainty = uncertainties_after[i].item()
        sentiment = "Positive" if pred >= 0.5 else "Negative"
        print(f"Text: \"{text}\"")
        print(f"  Prediction: {pred:.4f} ({sentiment})")
        print(f"  Uncertainty: {uncertainty:.4f}\n")
    print("-----------------------------------\n")

    # Optional: Save the model
    # torch.save(model.state_dict(), 'bayesian_nn_sentiment.pth')

if __name__ == "__main__":
    main()
