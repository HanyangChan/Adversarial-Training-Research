"""Load the exact upstream CIFAR ResNet without importing CUDA utilities."""
import importlib.util
from pathlib import Path

def build_model():
    path = Path(__file__).resolve().parents[1] / 'vendor/TDAT/models/resnet.py'
    spec = importlib.util.spec_from_file_location('tdat_upstream_resnet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ResNet18()

def forward_features(model, x):
    features = []
    handle = model.linear.register_forward_pre_hook(lambda _, args: features.append(args[0]))
    try:
        logits = model(x)
    finally:
        handle.remove()
    return logits, features[0]
