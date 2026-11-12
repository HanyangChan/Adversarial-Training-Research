"""Proposed loss, not part of TDAT. No persistent class statistics."""
import torch
import torch.nn.functional as F

def class_aware_feature_consistency(clean, adversarial, labels):
    # Equal weight per present class, then mean paired cosine distance.
    distances = 1 - F.cosine_similarity(clean.detach(), adversarial, dim=1)
    return torch.stack([distances[labels == c].mean() for c in labels.unique()]).mean()
