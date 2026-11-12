import argparse
import functools
import time
from pathlib import Path
import torch
import yaml
from .config import validate
from .model import build_model
from .data import loaders
from .attacks import accuracy, pgd
from .runtime import device_for, seed_all, metadata, write_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps'])
    p.add_argument('--attack',choices=['clean','pgd20','autoattack'],required=True)
    p.add_argument('--output',required=True); args=p.parse_args()
    config=validate(yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))); seed_all(config['seed']); device=device_for(args.device)
    _,test,_=loaders(config); model=build_model().to(device)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    state=checkpoint.get('model_state_dict',checkpoint)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()}); model.eval()
    start=time.perf_counter(); attack=None; e=config['eval']
    if args.attack=='pgd20':
        attack=functools.partial(pgd,epsilon=e['epsilon']/255,alpha=e['alpha']/255,
                                steps=e['pgd_steps'],restarts=e['restarts'])
    if args.attack=='autoattack':
        from autoattack import AutoAttack
        # One call on the full test set preserves standard AutoAttack orchestration.
        pairs=list(test); x=torch.cat([b[0] for b in pairs]).to(device); y=torch.cat([b[1] for b in pairs]).to(device)
        adversary=AutoAttack(model,norm='Linf',eps=e['epsilon']/255,version=e['autoattack_version'],device=str(device),seed=config['seed'])
        adv=adversary.run_standard_evaluation(x,y,bs=config['train']['batch_size'])
        correct=0
        with torch.no_grad():
            for a,b in zip(adv.split(config['train']['batch_size']),y.split(config['train']['batch_size'])):
                correct+=model(a).argmax(1).eq(b).sum().item()
        result=correct/len(y)
    else: result=accuracy(model,test,device,attack)
    info=metadata(config,device); info.update(attack=args.attack,accuracy=result,examples=len(test.dataset),
        seconds=time.perf_counter()-start,checkpoint=str(Path(args.checkpoint).resolve()))
    write_json(Path(args.output),info); print(info)

if __name__=='__main__': main()
