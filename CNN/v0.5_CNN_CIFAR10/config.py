"""Central configuration for CNN v0.5 on CIFAR-10."""

from pathlib import Path

SEED = 42
# Both v0.4 and v0.5 use the existing CIFAR-10 data in the parent CNN folder.
DATA_ROOT = str(Path(__file__).resolve().parent.parent)

TRAIN_BATCH_SIZE = 128
EVALUATION_BATCH_SIZE = 1000
VALIDATION_SIZE = 5_000

# Three residual stages; increase depth without making every layer very wide.
BASE_CHANNELS = 32
BLOCKS_PER_STAGE = (3, 3, 3)
# True: add shortcuts; False: use only each block's convolution branch.
USE_RESIDUAL = False
DROPOUT = 0.1

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.05

MAX_EPOCHS = 100
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 12
EARLY_STOPPING_MIN_DELTA = 1e-4

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3

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
