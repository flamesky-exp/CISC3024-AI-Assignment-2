"""AI-authored controlled IVON last-layer experiment. No test-based selection."""
from pathlib import Path
import os, sys, json, time, copy, random, math, platform, hashlib, argparse
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'vendor')]
import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from ivon import IVON
from starnet import create_model
from scipy.optimize import minimize_scalar

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def save(name,obj):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2),encoding='utf8')

def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

def metrics(p,y):
    p=np.asarray(p,dtype=np.float64);y=np.asarray(y)
    assert p.shape==(len(y),10) and np.isfinite(p).all()
    assert np.max(abs(p.sum(1)-1))<1e-5 and p.min()>=0
    pred=p.argmax(1);conf=p.max(1);correct=pred==y
    bins=np.minimum((conf*15).astype(int),14)
    rel=[];ece=0.
    for k in range(15):
        m=bins==k
        if m.any():
            a=float(correct[m].mean());c=float(conf[m].mean());n=int(m.sum())
            ece+=n/len(y)*abs(a-c);rel.append({'bin':k,'n':n,'accuracy':a,'confidence':c})
    return {'accuracy':float(correct.mean()),'nll':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),
            'brier':float(((p-np.eye(10)[y])**2).sum(1).mean()),'ece':ece,
            'entropy':float(-(p*np.log(np.clip(p,1e-12,1))).sum(1).mean()),'reliability':rel,
            'correct':int(correct.sum()),'n':len(y)}

@torch.no_grad()
def probs(head,x,opt=None,samples=1,draw_seed=991):
    # Preserve training RNG: validation sampling must not change later updates.
    if opt is None:return head(x).softmax(-1)
    with torch.random.fork_rng(devices=[0] if DEVICE=='cuda' else []):
        torch.manual_seed(draw_seed)
        out=torch.zeros((len(x),10),device=DEVICE)
        for _ in range(samples):
            with opt.sampled_params():out+=head(x).softmax(-1)
        return out/samples

def checks():
    seed(7);h=nn.Linear(3,10).to(DEVICE);o=IVON(h.parameters(),lr=.05,ess=100,hess_init=.1,beta2=.999,weight_decay=.001)
    before=[p.detach().clone() for p in h.parameters()]
    x=torch.randn(12,3,device=DEVICE);y=torch.arange(12,device=DEVICE)%10
    p=probs(h,x,o,8)
    assert all(torch.equal(a,b) for a,b in zip(before,h.parameters()))
    assert torch.allclose(p.sum(1),torch.ones(12,device=DEVICE),atol=1e-6)
    with o.sampled_params(train=True):
        o.zero_grad();nn.functional.cross_entropy(h(x),y).backward()
    o.step()
    assert all(torch.isfinite(v).all() for v in h.parameters())
    assert (o.param_groups[0]['hess']+.001>0).all()
    uniform=np.ones((10,10))/10
    z=metrics(uniform,np.arange(10))
    assert abs(z['nll']-math.log(10))<1e-10 and abs(z['brier']-.9)<1e-10 and z['ece']<1e-10
    perfect=metrics(np.eye(10),np.arange(10))
    assert perfect['accuracy']==1 and perfect['nll']==0 and perfect['brier']==0
    save('results/checks.json',{'posterior_restoration':True,'normalized_probabilities':True,'finite_update':True,'positive_precision':True,'metric_reference_cases':True})
    print('Meaningful numerical checks passed',flush=True)

def extract(data_root):
    cache=ROOT/'data/features.pt'
    if cache.exists():return torch.load(cache,map_location='cpu',weights_only=False)
    start=time.perf_counter();seed(42)
    m=create_model('starnet_s050').to(DEVICE).eval()
    ck=torch.load(ROOT/'checkpoints/backbone.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(ck['model_state'])
    tr=CIFAR10(data_root,train=True,download=False);te=CIFAR10(data_root,train=False,download=False)
    split=json.loads((ROOT/'data/splits/cifar10_seed42.json').read_text())
    a,b=split['train_indices'],split['validation_indices']
    assert len(a)==45000 and len(b)==5000 and not set(a)&set(b) and len(set(a+b))==50000
    means=torch.tensor([.4914,.4822,.4465],device=DEVICE)[None,:,None,None]
    std=torch.tensor([.2470,.2435,.2616],device=DEVICE)[None,:,None,None]
    d={}
    for name,ds,ix in [('train',tr,a),('val',tr,b),('test',te,list(range(10000)))]:
        fs=[];ls=[]
        with torch.inference_mode():
            for st in range(0,len(ix),256):
                ar=ds.data[np.asarray(ix[st:st+256])]
                x=torch.from_numpy(ar.copy()).permute(0,3,1,2).to(DEVICE,dtype=torch.float32)/255
                z,_=m.forward_features((x-means)/std);z=m.pool(m.norm(z)).flatten(1)
                fs.append(z.cpu());ls.append(m.head(z).cpu())
        d[name+'_x']=torch.cat(fs);d[name+'_y']=torch.tensor(np.asarray(ds.targets)[ix]);d[name+'_original_logits']=torch.cat(ls)
        print('Extracted',name,len(ix),flush=True)
    mu=d['train_x'].mean(0);sd=d['train_x'].std(0,unbiased=False).clamp_min(1e-6)
    for name in ['train','val','test']:d[name+'_x']=(d[name+'_x']-mu)/sd
    d['feature_mean']=mu;d['feature_std']=sd
    d['extraction_seconds']=time.perf_counter()-start
    torch.save(d,cache)
    save('results/features.json',{'extraction_seconds':d['extraction_seconds'],'dimensions':128,'backbone_parameters':sum(p.numel() for p in m.parameters()),'head_parameters':1290,'checkpoint_epoch':ck['epoch'],'checkpoint_validation_accuracy':ck['best_validation_accuracy']})
    return d

def make_head(s):
    seed(s);h=nn.Linear(128,10).to(DEVICE)
    nn.init.normal_(h.weight,std=.01);nn.init.zeros_(h.bias)
    return h

def fit(method,s,lr,d,cfg,epochs=None):
    epochs=epochs or cfg['epochs'];h=make_head(s)
    if method=='ivon':o=IVON(h.parameters(),lr=lr,ess=len(d['train_y']),hess_init=.1,beta1=.9,beta2=.999,weight_decay=.001,clip_radius=1.)
    else:o=torch.optim.Adam(h.parameters(),lr=lr,weight_decay=.001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs,eta_min=lr*.01)
    batchgen=torch.Generator(device=DEVICE).manual_seed(s+12345)
    best=float('inf');best_state=None;history=[];start=time.perf_counter()
    for ep in range(1,epochs+1):
        order=torch.randperm(len(d['train_y']),generator=batchgen,device=DEVICE)
        for ix in order.split(cfg['batch_size']):
            if method=='ivon':
                with o.sampled_params(train=True):
                    o.zero_grad();loss=nn.functional.cross_entropy(h(d['train_x'][ix]),d['train_y'][ix]);loss.backward()
            else:
                o.zero_grad();loss=nn.functional.cross_entropy(h(d['train_x'][ix]),d['train_y'][ix]);loss.backward()
            o.step()
        sched.step()
        if ep==1 or ep%5==0 or ep==epochs:
            p=probs(h,d['val_x'],o if method=='ivon' else None,cfg['validation_samples'])
            nll=float(nn.functional.nll_loss(p.clamp_min(1e-12).log(),d['val_y']))
            acc=float((p.argmax(1)==d['val_y']).float().mean())
            history.append({'epoch':ep,'val_nll':nll,'val_accuracy':acc})
            if nll<best:
                best=nll;best_state={'model':copy.deepcopy(h.state_dict()),'optimizer':copy.deepcopy(o.state_dict()),'epoch':ep,'val_nll':nll}
    if DEVICE=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    h.load_state_dict(best_state['model']);o.load_state_dict(best_state['optimizer'])
    result={'method':method,'seed':s,'lr':lr,'epoch':best_state['epoch'],'val_nll':best,'seconds':seconds,'history':history}
    tag=f'{method}_s{s}_lr{lr:g}'
    torch.save(best_state,ROOT/f'checkpoints/{tag}.pt');save(f'results/{tag}.json',result)
    print('FIT',tag,'best epoch',result['epoch'],'val NLL',round(best,5),'seconds',round(seconds,1),flush=True)
    return h,o,result

def load(method,s,lr):
    h=make_head(s)
    o=IVON(h.parameters(),lr=lr,ess=45000,hess_init=.1,beta2=.999,weight_decay=.001,clip_radius=1.) if method=='ivon' else torch.optim.Adam(h.parameters(),lr=lr,weight_decay=.001)
    ck=torch.load(ROOT/f'checkpoints/{method}_s{s}_lr{lr:g}.pt',map_location=DEVICE,weights_only=False)
    h.load_state_dict(ck['model']);o.load_state_dict(ck['optimizer'])
    return h,o

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['smoke','full','audit'],default='full');ap.add_argument('--data-root',default=r'D:\CISC3024 project 1\data\cifar10_stable');args=ap.parse_args()
    cfg={'epochs':60,'batch_size':1024,'seeds':[42,43,44],'adam_lr_grid':[.001,.01],'ivon_lr_grid':[.05,.2],
         'weight_decay':.001,'ivon_hess_init':.1,'ivon_beta2':.999,'ivon_clip_radius':1.,'ess':45000,'validation_samples':20,'test_samples':32,
         'selection':'seed-42 validation NLL; epochs 1,5,10,...,60; two learning rates per optimizer; no test selection',
         'protocol_date':'2026-10-11','feature_scaling':'training-only mean and population standard deviation','temperature_bounds':[.05,20.]}
    if not (ROOT/'results/protocol.json').exists():save('results/protocol.json',cfg)
    checks();d=extract(args.data_root)
    for k,v in d.items():
        if isinstance(v,torch.Tensor):d[k]=v.to(DEVICE)
    if args.stage=='smoke':
        sm=dict(d);sm['train_x']=d['train_x'][:2048];sm['train_y']=d['train_y'][:2048]
        for method,lr in [('adam',.001),('ivon',.05)]:fit(method,999,lr,sm,cfg,epochs=2)
        return
    if args.stage=='audit':
        selected=json.loads((ROOT/'results/selected.json').read_text());saved=np.load(ROOT/'results/predictions.npz');checks_out={}
        for s in cfg['seeds']:
            h,o=load('ivon',s,selected['ivon'])
            p=probs(h,d['test_x'],o,32,9000+s).cpu().numpy()
            delta=float(abs(p-saved[f'ivon_bma_{s}']).max());assert delta<1e-7
            checks_out[str(s)]=delta
        save('results/independent_audit.json',{'max_abs_probability_difference':checks_out,'passed':True})
        print('Independent reload audit passed',checks_out,flush=True);return
    selected={};runs={}
    for method in ['adam','ivon']:
        candidates=[]
        for lr in cfg[method+'_lr_grid']:
            h,o,r=fit(method,42,lr,d,cfg);candidates.append(r)
        winner=min(candidates,key=lambda z:z['val_nll']);selected[method]=winner['lr'];runs[(method,42)]=winner
        for s in [43,44]:
            h,o,r=fit(method,s,selected[method],d,cfg);runs[(method,s)]=r
    save('results/selected.json',selected) # Freeze before any test metric is computed.
    allrows=[];preds={};sample_rows=[]
    y=d['test_y'].cpu().numpy();preds['labels']=y
    for s in cfg['seeds']:
        h,o=load('adam',s,selected['adam'])
        with torch.no_grad():vl=h(d['val_x']);tl=h(d['test_x'])
        def obj(logt):return float(nn.functional.cross_entropy(vl/math.exp(logt),d['val_y']))
        ts=minimize_scalar(obj,bounds=(math.log(.05),math.log(20.)),method='bounded',options={'xatol':1e-6})
        t=math.exp(float(ts.x))
        ps={'adam':tl.softmax(-1),'adam_ts':(tl/t).softmax(-1)}
        ih,io=load('ivon',s,selected['ivon'])
        ps['ivon_mean']=probs(ih,d['test_x']);ps['ivon_bma']=probs(ih,d['test_x'],io,32,9000+s)
        for name,p in ps.items():
            arr=p.detach().cpu().numpy();preds[f'{name}_{s}']=arr
            r=metrics(arr,y);r.update(method=name,seed=s,temperature=t if name=='adam_ts' else 1.)
            r['fit_seconds']=runs[('ivon' if name.startswith('ivon') else 'adam',s)]['seconds'];allrows.append(r)
        if s==42:
            for n in [1,4,8,16,32,64]:
                p=probs(ih,d['test_x'],io,n,9000+s).cpu().numpy();r=metrics(p,y);r['samples']=n;sample_rows.append(r)
        # GPU timing after warm-up: head only, cached full-test features, 5 repetitions.
        for name in ps:
            timings=[]
            for rep in range(6):
                if DEVICE=='cuda':torch.cuda.synchronize()
                tick=time.perf_counter()
                if name=='ivon_bma':probs(ih,d['test_x'],io,32,9000+s)
                elif name=='ivon_mean':probs(ih,d['test_x'])
                else:
                    with torch.no_grad():(h(d['test_x'])/(t if name=='adam_ts' else 1)).softmax(-1)
                if DEVICE=='cuda':torch.cuda.synchronize()
                if rep:timings.append(time.perf_counter()-tick)
            next(r for r in allrows if r['seed']==s and r['method']==name)['head_inference_seconds']=float(np.median(timings))
    np.savez_compressed(ROOT/'results/predictions.npz',**preds)
    save('results/metrics.json',allrows);save('results/sample_sensitivity.json',sample_rows)
    save('results/original_backbone_reference.json',metrics(d['test_original_logits'].softmax(-1).cpu().numpy(),y))
    save('results/environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'device':DEVICE,'gpu':torch.cuda.get_device_name(0) if DEVICE=='cuda' else None,'deterministic':True})
    print('COMPLETE',selected,flush=True)
    for r in allrows:print(r['method'],r['seed'],{k:round(r[k],6) for k in ['accuracy','nll','brier','ece']},flush=True)

if __name__=='__main__':main()
