# UNet Implementation in PyTorch

This repository contains a modular implementation of the UNet model for image segmentation using PyTorch.

## Repository Structure
- `model.py`: Defines the UNet architecture.
- `layers.py`: Contains custom layer definitions (e.g., DoubleConv, Up).
- `utils.py`: Utility functions like weight initialization.
- `train.py`: Training and evaluation logic.
- `config.py`: Configuration parameters (e.g., hyperparameters).
- `main.py`: Entry point for running the model or training.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/unet_repository.git
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision
   ```

## Usage
1. Prepare your dataset and update `main.py` with the appropriate dataset class.
2. Adjust hyperparameters in `config.py` as needed.
3. Run the training script:
   ```bash
   python main.py
   ```

## Requirements
- Python 3.8+
- PyTorch 1.9+
- torchvision