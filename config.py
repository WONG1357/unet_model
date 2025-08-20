# Model configuration
IN_CHANNELS = 3
OUT_CHANNELS = 1
FEATURES = [64, 128, 256, 512]

# Training configuration
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"