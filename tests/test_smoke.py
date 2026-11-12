from pathlib import Path
import copy
import torch
import torch.nn.functional as F
import yaml
from tdat.model import build_model, forward_features
from tdat.core import initialize_delta, train_step
from tdat.consistency import class_aware_feature_consistency
from tdat.attacks import pgd, accuracy
from tdat.runtime import seed_all, device_for

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)

def test_real_resnet_training_and_checkpoint(tmp_path):
    config=yaml.safe_load((ROOT/'configs/cifar10_baseline.yaml').read_text())
    seed_all(config['seed'])
    x=torch.rand(2,3,32,32); y=torch.tensor([0,1])
    model=build_model(); initial=copy.deepcopy(model.state_dict())
    delta=initialize_delta(x,8/255,8/255)
    results=[]
    for enabled in [False,True]:
        model.load_state_dict(initial); model.train()
        config['feature_consistency']['enabled']=enabled
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        new_delta, losses=train_step(model,optimizer,x,y,delta.clone(),0,config)
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
        assert losses['feature']==0 if not enabled else losses['feature']>0
        assert not torch.equal(model.conv1.weight,initial['conv1.weight'])
        assert new_delta.abs().max()<=8/255+1e-6
        assert (x+new_delta).min()>=0 and (x+new_delta).max()<=1
        results.append(losses)
    model.eval()
    with torch.no_grad(): expected=model(x)
    checkpoint=tmp_path/'smoke.pt'
    torch.save({'model_state_dict':model.state_dict()},checkpoint)
    restored=build_model()
    restored.load_state_dict(torch.load(checkpoint,weights_only=True)['model_state_dict']); restored.eval()
    with torch.no_grad(): torch.testing.assert_close(restored(x),expected)
    assert expected.shape==(2,10)
    print('synthetic losses baseline / feature:',results)

def test_feature_class_balance_and_stop_gradient():
    clean=torch.tensor([[1.,0.],[1.,0.],[0.,1.]],requires_grad=True)
    adv=torch.tensor([[0.,1.],[1.,0.],[1.,0.]],requires_grad=True)
    y=torch.tensor([0,0,1])
    loss=class_aware_feature_consistency(clean,adv,y)
    torch.testing.assert_close(loss,torch.tensor(.75))
    loss.backward(); assert clean.grad is None; assert adv.grad is not None

def test_pgd20_and_clean_accuracy():
    seed_all(0)
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*32*32,10)).eval()
    x=torch.rand(3,3,32,32); y=model(x).argmax(1)
    adv=pgd(model,x,y,steps=20)
    assert adv.min()>=0 and adv.max()<=1
    assert (adv-x).abs().max()<=8/255+1e-6
    assert accuracy(model,[(x,y)],device_for('cpu'))==1
    assert 0<=accuracy(model,[(x,y)],device_for('cpu'),lambda m,a,b:pgd(m,a,b,steps=20))<=1

def test_original_feature_forward():
    model=build_model().eval(); x=torch.rand(2,3,32,32)
    with torch.no_grad():
        logits,features=forward_features(model,x)
        torch.testing.assert_close(logits,model(x))
    assert features.shape==(2,512)

def test_training_cli_synthetic_integration(tmp_path, monkeypatch):
    import json
    import sys
    from tdat import train, evaluate
    from torch.utils.data import DataLoader, TensorDataset
    config=yaml.safe_load((ROOT/'configs/cifar10_baseline.yaml').read_text())
    config['train'].update(epochs=1,batch_size=2)
    config['protocol']='synthetic_smoke_not_reproduction'
    config_path=tmp_path/'smoke.yaml'; config_path.write_text(yaml.safe_dump(config))
    x=torch.rand(2,3,32,32); y=torch.tensor([0,1])
    loader=DataLoader(TensorDataset(x,y),batch_size=2)
    monkeypatch.setattr(train,'loaders',lambda *a:(loader,loader,[0,1]))
    out=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['train','--config',str(config_path),'--device','cpu','--output',str(out)])
    train.main()
    metadata=__import__('json').loads((out/'metadata.json').read_text())
    assert metadata['training_seconds']>0 and metadata['cuda_peak_allocated_bytes'] is None
    assert metadata['config']['protocol']=='synthetic_smoke_not_reproduction'
    saved=torch.load(out/'epoch-000.pt',weights_only=True)
    assert saved['epoch']==0 and 'scheduler_state_dict' in saved
    monkeypatch.setattr(evaluate,'loaders',lambda *a:(loader,loader,[0,1]))
    result=tmp_path/'eval.json'
    monkeypatch.setattr(sys,'argv',['evaluate','--config',str(config_path),'--checkpoint',str(out/'epoch-000.pt'),
                                 '--device','cpu','--attack','pgd20','--output',str(result)])
    evaluate.main()
    assert 0<=json.loads(result.read_text())['accuracy']<=1

def test_configs_and_protocol_guard():
    import pytest
    from tdat.config import validate
    for path in (ROOT/'configs').glob('*.yaml'):
        validate(yaml.safe_load(path.read_text()))
    config=yaml.safe_load((ROOT/'configs/cifar10_baseline.yaml').read_text())
    config['train']['relaxation_function']='tanh'
    with pytest.raises(ValueError,match='separately named protocol'):
        validate(config)

def test_unborn_git_metadata(monkeypatch):
    import subprocess
    from tdat import runtime
    def fake_run(command, **kwargs):
        if command[1]=='rev-parse':
            return subprocess.CompletedProcess(command,128,stdout='HEAD\n',stderr='no commits')
        return subprocess.CompletedProcess(command,0,stdout='?? README.md\n',stderr='')
    monkeypatch.setattr(runtime.subprocess,'run',fake_run)
    info=runtime.metadata({'seed':0},torch.device('cpu'))
    assert info['git_commit']=='UNCOMMITTED' and info['git_dirty']
