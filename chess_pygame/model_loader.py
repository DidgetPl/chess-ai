import os

import torch
from evaluation_model import EvaluationModel

MODEL_DIR = ".\\training\\checkpoints"

def list_available_models():
    models = []
    if not os.path.isdir(MODEL_DIR):
        return models

    for folder in sorted(os.listdir(MODEL_DIR)):
        folder_path = os.path.join(MODEL_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        for f in os.listdir(folder_path):
            if f.endswith(".ckpt"):
                relative_path = os.path.join(folder, f)
                models.append(relative_path)

    return sorted(models)


def load_model(filename, device="cpu"):
    path = os.path.join(MODEL_DIR, filename)
    model = EvaluationModel.load_from_checkpoint(path)
    model.eval()
    model.to(device)
    return model
