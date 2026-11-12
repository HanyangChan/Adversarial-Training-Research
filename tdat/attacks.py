import torch
import torch.nn.functional as F

def pgd(model, x, y, epsilon=8/255, alpha=2/255, steps=20, restarts=1):
    """Upstream CE PGD: update only currently correctly classified samples."""
    best_loss = x.new_zeros(len(x))
    best_delta = torch.zeros_like(x)
    for _ in range(restarts):
        d = torch.empty_like(x).uniform_(-epsilon, epsilon)
        d = torch.maximum(torch.minimum(d, 1-x), -x)
        for _ in range(steps):
            d = d.detach().requires_grad_(True)
            logits = model(x+d)
            active = logits.argmax(1).eq(y)
            if not active.any():
                break
            gradient, = torch.autograd.grad(F.cross_entropy(logits, y), d)
            update = (d + alpha*gradient.sign()).clamp(-epsilon, epsilon)
            update = torch.maximum(torch.minimum(update, 1-x), -x)
            d = torch.where(active[:,None,None,None], update, d).detach()
        with torch.no_grad():
            losses = F.cross_entropy(model(x+d), y, reduction='none')
            best_delta[losses >= best_loss] = d.detach()[losses >= best_loss]
            best_loss = torch.maximum(losses, best_loss)
    return (x+best_delta).detach()

def accuracy(model, loader, device, attack=None):
    model.eval()
    correct = total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        if attack:
            x = attack(model, x, y)
        with torch.no_grad():
            correct += model(x).argmax(1).eq(y).sum().item()
        total += len(y)
    if not total:
        raise ValueError('Empty evaluation set')
    return correct / total
