import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from train import train_model, evaluate_model
from utils import apply_init_weights
from config import IN_CHANNELS, OUT_CHANNELS, FEATURES, BATCH_SIZE, LEARNING_RATE, EPOCHS, DEVICE
# Assume a dataset is available (e.g., your custom dataset)
from dataset import CustomDataset  # Placeholder, replace with actual dataset

def main():
    # Initialize model
    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, features=FEATURES)
    model = apply_init_weights(model)
    model = model.to(DEVICE)

    # Placeholder dataset and dataloaders
    train_dataset = CustomDataset(...)  # Replace with actual dataset
    test_dataset = CustomDataset(...)   # Replace with actual dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train and evaluate
    model = train_model(model, train_loader, criterion, optimizer, DEVICE, EPOCHS)
    evaluate_model(model, test_loader, criterion, DEVICE)

if __name__ == "__main__":
    main()