import json
import platform
import random
import subprocess
import numpy as np
import torch

def device_for(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable')
    return torch.device(name)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def metadata(config, device):
    commit = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD^{commit}'], capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    return dict(config=config, seed=config['seed'], git_commit=(commit.stdout.strip() if commit.returncode == 0 else 'UNCOMMITTED'),
        git_dirty=bool(dirty.stdout.strip()), upstream_commit='2ce7bb0650584756a8bbb8deca575da4150718c4',
        python=platform.python_version(), platform=platform.platform(), torch=str(torch.__version__),
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        gpu_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None)

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def synchronize(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)
    elif device.type == 'mps': torch.mps.synchronize()
