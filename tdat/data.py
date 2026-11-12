import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def loaders(config, download=False):
    c = config['data']
    augmentation = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    train = datasets.CIFAR10(c['root'], train=True, download=download, transform=augmentation)
    test = datasets.CIFAR10(c['root'], train=False, download=download, transform=transforms.ToTensor())
    fraction = c['train_fraction']
    if not 0 < fraction <= 1:
        raise ValueError('train_fraction must be in (0,1]')
    indices = list(range(len(train)))
    if fraction < 1:
        rng = np.random.default_rng(config['seed'])
        targets = np.asarray(train.targets)
        indices = sorted(np.concatenate([rng.permutation(np.where(targets == k)[0])[:int(5000*fraction)]
                                         for k in range(10)]).tolist())
        train = Subset(train, indices)
    kwargs = dict(batch_size=config['train']['batch_size'], num_workers=c['workers'])
    # As upstream: shuffle both sets, retain partial batch but training skips it.
    return DataLoader(train, shuffle=True, **kwargs), DataLoader(test, shuffle=True, **kwargs), indices
