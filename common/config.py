
CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

NUM_CLASSES = len(CLASSES)
IMAGE_SIZE = 224
BATCH_SIZE = 16
LOCAL_EPOCHS = 2
NUM_ROUNDS = 3

SERVER_ADDRESS = "0.0.0.0:8080"
