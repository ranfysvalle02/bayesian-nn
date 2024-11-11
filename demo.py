import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Constants
INPUT_DIM = 100
HIDDEN_DIM = 50  # Increased from 10 to 50
OUTPUT_DIM = 1
DROPOUT_PROB = 0.5  # Increased from 0.1 to 0.5
NUM_SAMPLES = 1000
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 10
MC_SAMPLES = 100  # Number of Monte Carlo samples for uncertainty

# Synthetic Text Dataset Generation
class SentimentDataset(Dataset):
    """A synthetic dataset for sentiment analysis."""

    def __init__(self, num_samples=1000):
        self.texts = [
            "I love this product!" if i % 2 == 0 else "This is terrible."
            for i in range(num_samples)
        ]
        self.labels = [1 if i % 2 == 0 else 0 for i in range(num_samples)]
        self.tokenizer = self.char_tokenizer

    @staticmethod
    def char_tokenizer(text):
        """Simple character-based tokenizer converting characters to ASCII codes."""
        return [ord(char) for char in text]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized_text = self.tokenizer(text)
        return torch.tensor(tokenized_text, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def collate_fn(batch, max_length=INPUT_DIM):
    """
    Pads or truncates text sequences to a fixed length.

    Args:
        batch: A list of tuples (tokenized_text, label).
        max_length: The fixed length to pad/truncate the sequences.

    Returns:
        Padded texts tensor and labels tensor.
    """
    texts, labels = zip(*batch)
    padded_texts = []
    for text in texts:
        if len(text) < max_length:
            padded = torch.cat([text, torch.zeros(max_length - len(text))])
        else:
            padded = text[:max_length]
        padded_texts.append(padded)
    return torch.stack(padded_texts), torch.tensor(labels, dtype=torch.float32)

# Bayesian Neural Network using Monte Carlo Dropout
class BayesianNN(pl.LightningModule):
    """A simple Bayesian Neural Network for binary classification."""

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, p=DROPOUT_PROB):
        super(BayesianNN, self).__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        """Forward pass through the network."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.output_activation(x)
        return x

    def training_step(self, batch, batch_idx):
        """Defines the training step."""
        x, y = batch
        y_pred = self(x).squeeze()
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configures the optimizer."""
        optimizer = Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer

def make_prediction(model, x, text, num_samples=MC_SAMPLES):
    """
    Makes predictions using Monte Carlo Dropout to estimate uncertainty.

    Args:
        model: The trained BayesianNN model.
        x: Input tensor.
        text: The text being predicted (for print statements).
        num_samples: Number of forward passes to perform.

    Returns:
        Mean prediction and uncertainty (standard deviation).
    """
    model.train()  # Enable dropout
    predictions = []
    print(f"Predicting sentiment for: '{text}'")
    with torch.no_grad():
        for i in range(num_samples):
            pred = model(x)
            predictions.append(pred)
    predictions = torch.stack(predictions)
    mean_prediction = torch.mean(predictions, dim=0)
    uncertainty = torch.std(predictions, dim=0)
    return mean_prediction, uncertainty

def main():
    """Main function to execute the training, prediction, and feedback process."""

    # Generate synthetic dataset
    dataset = SentimentDataset(num_samples=NUM_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Display sample data from the dataset
    print("\n--- Sample Data from SentimentDataset ---")
    for i in range(3):
        sample_text = dataset.texts[i]
        sample_label = dataset.labels[i]
        print(f"Sample {i+1}:")
        print(f"  Text: {sample_text}")
        print(f"  Label: {'Positive' if sample_label == 1 else 'Negative'}")
    print("------------------------------------------\n")

    # Initialize and train the model
    model = BayesianNN(input_dim=INPUT_DIM)

    # Use the updated TQDMProgressBar callback to set refresh rate
    progress_bar = TQDMProgressBar(refresh_rate=20)  # Set your desired refresh rate here
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=False,
        enable_checkpointing=False,
        callbacks=[progress_bar]
    )

    print("Starting training...")
    trainer.fit(model, dataloader)
    print("Training completed.\n")

    # Prepare test input
    test_text = "This product is amazing!"
    print("--- Test Input ---")
    print(f"Text: {test_text}")
    x_test = torch.tensor([ord(char) for char in test_text], dtype=torch.float32)
    if len(x_test) < INPUT_DIM:
        x_test = torch.cat([x_test, torch.zeros(INPUT_DIM - len(x_test))])
    else:
        x_test = x_test[:INPUT_DIM]
    print(f"Tokenized and padded input: {x_test}\n")
    print("-------------------\n")

    # Make a prediction before feedback
    print("Making prediction before feedback...")
    mean_pred, uncertainty = make_prediction(model, x_test.unsqueeze(0), test_text)
    print(f"Prediction before feedback: {mean_pred.item():.4f}")
    print(f"Uncertainty: {uncertainty.item():.4f}\n")

    # Display initial prediction
    sentiment_before = "Positive" if mean_pred.item() >= 0.5 else "Negative"
    print(f"Initial Sentiment Prediction: {sentiment_before} (Confidence: {mean_pred.item():.2f})")
    print(f"Prediction Uncertainty: {uncertainty.item():.2f}\n")

    # Feedback - retrain with additional positive data
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
        "Top-notch service."
    ]
    feedback_labels = [1] * len(feedback_texts)  # All positive labels

    print("--- Feedback Data ---")
    for i, text in enumerate(feedback_texts):
        print(f"Feedback {i+1}:")
        print(f"  Text: {text}")
        print(f"  Label: {'Positive' if feedback_labels[i] == 1 else 'Negative'}")
    print("---------------------\n")

    # Tokenize and pad feedback data
    x_feedback = [torch.tensor([ord(char) for char in text], dtype=torch.float32) for text in feedback_texts]
    x_feedback = [
        torch.cat([x, torch.zeros(INPUT_DIM - len(x))]) if len(x) < INPUT_DIM else x[:INPUT_DIM]
        for x in x_feedback
    ]
    y_feedback = torch.tensor(feedback_labels, dtype=torch.float32)
    feedback_dataset = TensorDataset(torch.stack(x_feedback), y_feedback)
    feedback_dataloader = DataLoader(
        feedback_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.stack([item[1] for item in batch])
        )
    )

    # Retrain the model with feedback data
    print("Incorporating feedback and retraining the model...")
    trainer.fit(model, feedback_dataloader)
    print("Retraining with feedback completed.\n")

    # Make a prediction after feedback
    print("Making prediction after feedback...")
    mean_pred_after, uncertainty_after = make_prediction(model, x_test.unsqueeze(0), test_text)
    print(f"Prediction after feedback: {mean_pred_after.item():.4f}")
    print(f"Uncertainty: {uncertainty_after.item():.4f}\n")

    # Display updated prediction
    sentiment_after = "Positive" if mean_pred_after.item() >= 0.5 else "Negative"
    print(f"Updated Sentiment Prediction: {sentiment_after} (Confidence: {mean_pred_after.item():.2f})")
    print(f"Prediction Uncertainty: {uncertainty_after.item():.2f}\n")

if __name__ == "__main__":
    main()
