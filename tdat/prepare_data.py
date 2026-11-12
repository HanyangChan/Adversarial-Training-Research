"""학습 없이 공식 CIFAR-10 파일과 데이터 경로를 검증합니다."""
import argparse
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity
import yaml
from .config import validate
from .data import loaders
from .runtime import seed_all, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/cifar10_baseline.yaml')
    parser.add_argument('--download', action='store_true', help='공식 파일이 없으면 다운로드')
    parser.add_argument('--output', default='runs/data-validation.json')
    args = parser.parse_args()
    config = validate(yaml.safe_load(Path(args.config).read_text(encoding='utf-8')))
    if config['data']['train_fraction'] != 1.0:
        raise ValueError('전체 데이터 검증에는 train_fraction=1.0 설정을 사용하세요')
    seed_all(config['seed'])
    if args.download:
        CIFAR10(config['data']['root'], train=True, download=True)
    archive = Path(config['data']['root']) / CIFAR10.filename
    if not check_integrity(archive, CIFAR10.tgz_md5):
        raise ValueError(f'공식 압축파일 MD5 검증 실패 또는 파일 없음: {archive}')
    train, test, indices = loaders(config, download=False)
    train_counts = np.bincount(train.dataset.targets, minlength=10).tolist()
    test_counts = np.bincount(test.dataset.targets, minlength=10).tolist()
    assert len(train.dataset) == 50000 and len(test.dataset) == 10000
    assert train_counts == [5000]*10 and test_counts == [1000]*10
    assert indices == list(range(50000))
    batches = {}
    for name, loader in [('train', train), ('test', test)]:
        x, y = next(iter(loader))
        assert x.shape == (config['train']['batch_size'], 3, 32, 32)
        assert x.dtype == torch.float32 and y.dtype == torch.int64
        assert torch.isfinite(x).all() and 0 <= x.min() <= x.max() <= 1
        assert 0 <= y.min() <= y.max() < 10
        batches[name] = dict(shape=list(x.shape), dtype=str(x.dtype), minimum=float(x.min()), maximum=float(x.max()))
    classes = train.dataset.classes
    del train, test
    reduced = copy.deepcopy(config)
    reduced['data']['train_fraction'] = .1
    train, test, sampled = loaders(reduced)
    targets = np.asarray(train.dataset.dataset.targets)[sampled]
    reduced_counts = np.bincount(targets, minlength=10).tolist()
    assert reduced_counts == [500]*10 and len(test.dataset) == 10000
    # 데이터셋 재로딩 없이 같은 seed의 표본 인덱스를 재확인합니다.
    rng = np.random.default_rng(config['seed'])
    all_targets = np.asarray(train.dataset.dataset.targets)
    expected = sorted(np.concatenate([rng.permutation(np.where(all_targets == k)[0])[:500] for k in range(10)]).tolist())
    assert sampled == expected and len(set(sampled)) == 5000
    result = dict(source=CIFAR10.url, final_download_source='https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz',
        archive=str(archive.resolve()), md5=CIFAR10.tgz_md5, md5_verified=True,
        archive_bytes=archive.stat().st_size, train_examples=50000, test_examples=10000,
        train_class_counts=train_counts, test_class_counts=test_counts, classes=classes,
        batches=batches, reduced_fraction=.1, reduced_examples=5000, reduced_class_counts=reduced_counts,
        sampling_seed=config['seed'], sampled_indices_sha256=hashlib.sha256(np.array(sampled,dtype='<i8').tobytes()).hexdigest(),
        training_executed=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(result)


if __name__ == '__main__':
    main()
