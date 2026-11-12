import math
import torch
import torch.nn.functional as F
from .model import forward_features
from .consistency import class_aware_feature_consistency

def initialize_delta(x, epsilon, alpha):
    return (torch.empty_like(x).uniform_(-epsilon, epsilon).sign() * alpha).clamp(-epsilon, epsilon)

def train_step(model, optimizer, x, y, delta, epoch, config):
    """Public-code semantics including tan, outer smoothing and momentum overwrite."""
    c = config['train']
    epsilon, alpha = c['epsilon']/255, c['alpha']/255
    gamma = max(math.tan(1-epoch/c['epochs'])*c['beta'], c['inner_gamma'])
    outer = c['outer_gamma'] if math.tan(1-epoch/c['epochs'])*c['beta'] < c['inner_gamma'] else gamma
    delta = delta.detach().requires_grad_(True)
    logits = model(x + delta)  # upstream does NOT clamp the initial x+delta
    soft = F.one_hot(y, 10).to(x.dtype)*gamma
    soft += (1-F.one_hot(y, 10).to(x.dtype))*((1-gamma)/9)
    inner = F.cross_entropy(logits, soft)
    inner.backward(retain_graph=True)
    adversarial_delta = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon, epsilon)
    adversarial_delta = torch.maximum(torch.minimum(adversarial_delta, 1-x), -x).detach()
    output, adv_features = forward_features(model, x + adversarial_delta)
    ce = F.cross_entropy(output, y, label_smoothing=1-outer)
    p = logits.softmax(1).gather(1, y[:,None]).squeeze(1)
    robust = ((logits-output).square().sum(1)*torch.tanh(1.0000001-p)).mean()
    loss = ce + c['lamda']*robust
    consistency = x.new_zeros(())
    if config['feature_consistency']['enabled']:
        # Extra clean pass only when enabled; eval avoids modifying BN statistics.
        model.eval()
        with torch.no_grad():
            _, clean_features = forward_features(model, x)
        model.train()
        consistency = class_aware_feature_consistency(clean_features, adv_features, y)
        loss = loss + config['feature_consistency']['weight']*consistency
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Deliberately retain upstream overwrite, rather than repairing batch momentum.
    return adversarial_delta, {k: float(v.detach()) for k,v in
        dict(loss=loss, inner=inner, ce=ce, robust=robust, feature=consistency).items()}
