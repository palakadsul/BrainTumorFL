
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4
IMAGE_SIZE = 224

def create_model(num_classes=NUM_CLASSES):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_parameters(model):
    return [
        value.cpu().numpy()
        for value in model.state_dict().values()
    ]

def set_parameters(model, parameters):
    state_dict = model.state_dict()
    keys = list(state_dict.keys())

    import torch

    new_state_dict = {
        key: torch.tensor(value, dtype=state_dict[key].dtype)
        for key, value in zip(keys, parameters)
    }

    model.load_state_dict(new_state_dict, strict=True)
