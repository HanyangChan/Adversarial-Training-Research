import argparse
import functools
import time
from pathlib import Path
import torch
import yaml
from .config import validate
from .model import build_model
from .core import initialize_delta, train_step
from .data import loaders
from .attacks import accuracy, pgd
from .runtime import device_for, seed_all, metadata, write_json, synchronize

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    p.add_argument('--download', action='store_true')
    p.add_argument('--output', required=True)
    args=p.parse_args()
    config=validate(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    c=config['train']; device=device_for(args.device); seed_all(config['seed'])
    out=Path(args.output); out.mkdir(parents=True, exist_ok=False)
    train,test,indices=loaders(config,args.download)
    write_json(out/'train_indices.json', indices)
    model=build_model().to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=c['lr'],momentum=c['momentum'],weight_decay=c['weight_decay'])
    # Upstream scheduler uses ceil(50000/128) although partial batches are skipped.
    nominal_batches=(len(indices)+c['batch_size']-1)//c['batch_size']
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=[nominal_batches*k for k in c['milestones']],gamma=c['lr_gamma'])
    delta=initialize_delta(torch.empty(c['batch_size'],3,32,32,device=device),c['epsilon']/255,c['alpha']/255)
    info=metadata(config,device); write_json(out/'metadata.json',info)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    synchronize(device); start=time.perf_counter(); history=[]
    for epoch in range(c['epochs']):
        model.train(); synchronize(device); epoch_start=time.perf_counter(); count=0; loss_sum=0
        for x,y in train:
            if len(y)!=c['batch_size']: continue
            delta,losses=train_step(model,optimizer,x.to(device),y.to(device),delta,epoch,config)
            scheduler.step(); count+=len(y); loss_sum+=losses['loss']*len(y)
        if count==0: raise ValueError('No full training batches')
        synchronize(device); train_seconds=time.perf_counter()-epoch_start
        clean=accuracy(model,test,device)
        attack=functools.partial(pgd,steps=config['eval']['epoch_pgd_steps'],restarts=config['eval']['restarts'],
            epsilon=config['eval']['epsilon']/255,alpha=config['eval']['alpha']/255)
        robust=accuracy(model,test,device,attack)
        row=dict(epoch=epoch,train_seconds=train_seconds,loss=loss_sum/count,train_examples=count,
            clean_accuracy=clean,pgd10_accuracy=robust,lr=scheduler.get_last_lr()[0])
        history.append(row); write_json(out/'history.json',history); print(row,flush=True)
        if epoch>=c['save_from_epoch'] or epoch==c['epochs']-1:
            torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),epoch=epoch,metadata=info),out/f'epoch-{epoch:03d}.pt')
    synchronize(device)
    info.update(total_seconds=time.perf_counter()-start,training_seconds=sum(r['train_seconds'] for r in history),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
    write_json(out/'metadata.json',info)

if __name__=='__main__': main()
