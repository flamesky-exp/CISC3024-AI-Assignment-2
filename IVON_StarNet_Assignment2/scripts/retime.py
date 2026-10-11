"""Comparable head-only inference timing; no retraining or model selection."""
import experiment as e
import json, time, numpy as np, torch
rows=json.loads((e.ROOT/'results/metrics.json').read_text())
selected=json.loads((e.ROOT/'results/selected.json').read_text())
d=torch.load(e.ROOT/'data/features.pt',map_location='cpu',weights_only=False)
x=d['test_x'].to(e.DEVICE)
for s in [42,43,44]:
    ah,ao=e.load('adam',s,selected['adam']);ih,io=e.load('ivon',s,selected['ivon'])
    for name in ['adam','adam_ts','ivon_mean','ivon_bma']:
        row=next(r for r in rows if r['method']==name and r['seed']==s);t=row['temperature'];times=[]
        for i in range(25):
            if e.DEVICE=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            with torch.no_grad():
                if name=='ivon_bma':e.probs(ih,x,io,32,9000+s)
                elif name=='ivon_mean':e.probs(ih,x)
                else:(ah(x)/t).softmax(-1)
            if e.DEVICE=='cuda':torch.cuda.synchronize()
            if i>=5:times.append(time.perf_counter()-start)
        row['head_inference_seconds']=float(np.median(times))
e.save('results/metrics.json',rows)
print('Comparable inference timing: 5 warm-ups, median of 20 repeats per seed.')
