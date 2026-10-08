"""Central configuration for CNN v0.4 on CIFAR-10."""

SEED = 42
DATA_ROOT = ""

TRAIN_BATCH_SIZE = 128
EVALUATION_BATCH_SIZE = 1000
VALIDATION_SIZE = 5_000

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.05

MAX_EPOCHS = 100
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 12
EARLY_STOPPING_MIN_DELTA = 1e-4

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3

DROPOUT = 0.2

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
