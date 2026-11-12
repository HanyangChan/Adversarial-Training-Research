"""Reject settings for which this port has no faithful implementation."""
def validate(config):
    c=config['train']
    fixed={'delta_init':'random','relaxation_function':'tan',
           'preserve_momentum_overwrite':True,'skip_partial_batch':True}
    for key,value in fixed.items():
        if c.get(key)!=value:
            raise ValueError(f'{key} must be {value!r}; create a separately named protocol implementation for changes')
    if config.get('model')!='ResNet18':
        raise ValueError('This port implements only the upstream CIFAR ResNet18')
    if c['epochs']<1 or c['batch_size']<1 or not 0<config['data']['train_fraction']<=1:
        raise ValueError('Invalid epochs, batch size, or sampling fraction')
    if c['epsilon']<=0 or c['alpha']<=0 or config['eval']['restarts']<1:
        raise ValueError('Attack budgets and restart count must be positive')
    return config
