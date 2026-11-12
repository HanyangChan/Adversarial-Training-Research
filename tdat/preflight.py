"""실제 CIFAR 소량으로 환경·모델을 진단합니다. 성능 재현 실험이 아닙니다."""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path


def run(args):
    import torch
    import torch.nn.functional as F
    import yaml
    from .attacks import pgd
    from .config import validate
    from .core import initialize_delta, train_step
    from .data import loaders
    from .model import build_model
    from .runtime import device_for, metadata, seed_all, synchronize, write_json

    start = time.perf_counter()
    torch.set_num_threads(args.threads)
    config = validate(yaml.safe_load(Path(args.config).read_text(encoding='utf-8')))
    config = copy.deepcopy(config)
    config['train']['batch_size'] = args.batch_size
    config['protocol'] = 'real_cifar_preflight_not_baseline_reproduction'
    device = device_for(args.device)
    seed_all(config['seed'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    info = metadata(config, device)
    info.update(status='running', diagnostic_steps_per_variant=args.steps,
                source_config=args.config, purpose='environment_and_model_test_only')
    write_json(out/'result.json', info)
    train, test, _ = loaders(config, download=False)
    x, y = next(iter(train))
    tx, ty = next(iter(test))
    x, y, tx, ty = (v.to(device) for v in (x, y, tx, ty))
    assert len(train.dataset) == 50000 and len(test.dataset) == 10000
    assert x.shape == (args.batch_size, 3, 32, 32)
    assert torch.isfinite(x).all() and 0 <= x.min() <= x.max() <= 1
    model = build_model().to(device)
    initial = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    for enabled in (False, True):
        model.load_state_dict(initial)
        seed_all(config['seed'])
        config['feature_consistency']['enabled'] = enabled
        c = config['train']
        optimizer = torch.optim.SGD(model.parameters(), lr=c['lr'], momentum=c['momentum'], weight_decay=c['weight_decay'])
        delta = initialize_delta(x, c['epsilon']/255, c['alpha']/255)
        model.train()
        synchronize(device)
        begin = time.perf_counter()
        losses = []
        for step in range(args.steps):
            delta, row = train_step(model, optimizer, x, y, delta, 0, config)
            assert all(torch.isfinite(torch.tensor(v)) for v in row.values())
            assert delta.abs().max() <= c['epsilon']/255+1e-6
            assert (x+delta).min() >= 0 and (x+delta).max() <= 1
            losses.append(row)
        synchronize(device)
        seconds = time.perf_counter()-begin
        assert not torch.equal(model.conv1.weight.detach().cpu(), initial['conv1.weight'])
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            output = model(tx)
            assert output.shape == (args.batch_size, 10)
            assert torch.isfinite(F.cross_entropy(output, ty))
        record = dict(feature_enabled=enabled, steps=args.steps, seconds=seconds, losses=losses)
        records.append(record)
        print(json.dumps(record), flush=True)
    # Untrained diagnostic weights: no accuracy/robustness claims.
    e = config['eval']
    synchronize(device)
    attack_start = time.perf_counter()
    adv = pgd(model, tx, ty, epsilon=e['epsilon']/255, alpha=e['alpha']/255,
              steps=e['pgd_steps'], restarts=e['restarts'])
    synchronize(device)
    assert torch.isfinite(adv).all() and adv.min() >= 0 and adv.max() <= 1
    assert (adv-tx).abs().max() <= e['epsilon']/255+1e-6
    attack_seconds = time.perf_counter()-attack_start
    checkpoint = out/'diagnostic.pt'
    torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    config=config, purpose='diagnostic_not_trained_baseline'), checkpoint)
    restored = build_model().to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model_state_dict'])
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(tx), output)
    synchronize(device)
    info.update(status='passed', variants=records, elapsed_seconds=time.perf_counter()-start,
        train_examples_inspected=len(y), test_examples_inspected=len(ty),
        pgd_requested_steps=e['pgd_steps'], pgd_seconds=attack_seconds,
        pgd_note='upstream early-stop; may execute fewer steps on diagnostic weights',
        checkpoint_roundtrip=True, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
    # 최종 활성화 상태를 기록하고 각 비교 조건은 variants에 별도로 보존합니다.
    info['config'] = copy.deepcopy(config)
    write_json(out/'result.json', info)
    print('PASS:', out/'result.json', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/cifar10_baseline.yaml')
    p.add_argument('--device', choices=['cpu','cuda','mps'], default='cpu')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--max-seconds', type=int, default=900)
    p.add_argument('--output', required=True)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if min(args.batch_size,args.steps,args.threads,args.max_seconds) < 1:
        p.error('batch-size, steps, threads, max-seconds must be positive')
    if args.worker:
        run(args)
        return
    if Path(args.output).exists():
        p.error('output must be a new directory')
    try:
        result = subprocess.run([sys.executable,'-m','tdat.preflight',*sys.argv[1:],'--worker'], timeout=args.max_seconds)
    except subprocess.TimeoutExpired:
        path = Path(args.output)
        path.mkdir(parents=True,exist_ok=True)
        (path/'timeout.json').write_text(json.dumps(dict(status='timeout',max_seconds=args.max_seconds)))
        raise SystemExit(124)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
