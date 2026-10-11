"""Generate tables/figures and conditional paired bootstrap intervals from saved predictions."""
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results';F=ROOT/'report/figures';F.mkdir(parents=True,exist_ok=True)
rows=json.loads((R/'metrics.json').read_text());a=np.load(R/'predictions.npz');y=a['labels']
methods=['adam','adam_ts','ivon_mean','ivon_bma']
labels=['Adam','Adam + temperature','IVON mean','IVON BMA (32)']
colors=['#516987','#3a9a9c','#b47d43','#8a529a']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight'})
summary={}
for m in methods:
    rr=[r for r in rows if r['method']==m];summary[m]={}
    for k in ['accuracy','nll','brier','ece','fit_seconds','head_inference_seconds']:
        v=np.array([r[k] for r in rr]);summary[m][k]={'mean':float(v.mean()),'sd':float(v.std(ddof=1))}
with (R/'summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['method','accuracy_mean','accuracy_sd','nll_mean','nll_sd','brier_mean','brier_sd','ece_mean','ece_sd'])
    for m in methods:w.writerow([m]+[summary[m][k][t] for k in ['accuracy','nll','brier','ece'] for t in ['mean','sd']])
(R/'summary.json').write_text(json.dumps(summary,indent=2))
fig,axs=plt.subplots(2,2,figsize=(8.4,6.3))
for ax,k,title,scale in zip(axs.flat,['accuracy','nll','brier','ece'],['Accuracy (%) - higher is better','NLL - lower is better','Brier score - lower is better','ECE (%) - lower is better'],[100,1,1,100]):
    vals=[summary[m][k]['mean']*scale for m in methods];sd=[summary[m][k]['sd']*scale for m in methods]
    ax.bar(range(4),vals,yerr=sd,capsize=4,color=colors,width=.6)
    ax.set_xticks(range(4),['Adam','Adam+TS','IVON\nmean','IVON\nBMA']);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,f'{v:.3f}' if scale==1 else f'{v:.2f}',ha='center',fontsize=8)
    ax.set_ylim(0,max(vals)*1.2)
fig.tight_layout();fig.savefig(F/'metric_comparison.pdf');fig.savefig(F/'metric_comparison.png');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(8.2,6.5))
for ax,m,label,color in zip(axs.flat,methods,labels,colors):
    r=next(r for r in rows if r['method']==m and r['seed']==42);rel=r['reliability']
    ax.plot([0,1],[0,1],'--',c='#999999',lw=1)
    ax.scatter([b['confidence'] for b in rel],[b['accuracy'] for b in rel],s=[20+180*b['n']/10000 for b in rel],color=color)
    ax.plot([b['confidence'] for b in rel],[b['accuracy'] for b in rel],c=color,lw=1)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence',ylabel='Empirical accuracy',title=f'{label}: ECE {100*r["ece"]:.2f}%')
    ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(F/'reliability.pdf');fig.savefig(F/'reliability.png');plt.close(fig)
sens=json.loads((R/'sample_sensitivity.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(8.3,3.2))
for ax,k,scale,title in [(axs[0],'nll',1,'NLL'),(axs[1],'ece',100,'ECE (%)')]:
    ax.plot([r['samples'] for r in sens],[r[k]*scale for r in sens],'-o',color=colors[3]);ax.set_xscale('log',base=2);ax.set_xticks([1,4,8,16,32,64],[1,4,8,16,32,64]);ax.set(xlabel='Posterior draws',ylabel=title);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(F/'sample_sensitivity.pdf');fig.savefig(F/'sample_sensitivity.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.5,3.6));risk={}
for m,label,color in zip(methods,labels,colors):
    p=a[f'{m}_42'];order=np.argsort(-p.max(1),kind='stable');errs=(p.argmax(1)!=y)[order];cov=np.arange(1,len(y)+1)/len(y);rr=errs.cumsum()/np.arange(1,len(y)+1)
    ax.plot(cov[99::100]*100,rr[99::100]*100,label=label,c=color);risk[m]={str(c):float(rr[int(c*len(y))-1]) for c in [.5,.8,1.]}
ax.set(xlabel='Coverage: fraction retained (%)',ylabel='Error rate among retained images (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(F/'risk_coverage.pdf');fig.savefig(F/'risk_coverage.png');plt.close(fig)
(R/'selective_risk.json').write_text(json.dumps(risk,indent=2))
# Bootstrap images; seeds are averaged within each image, not falsely treated as 30k independent samples.
rng=np.random.default_rng(20261011);boot={}
for baseline in ['adam','adam_ts','ivon_mean']:
    diff=np.mean([(-np.log(a[f'ivon_bma_{s}'][np.arange(len(y)),y].clip(1e-12))-(-np.log(a[f'{baseline}_{s}'][np.arange(len(y)),y].clip(1e-12)))) for s in [42,43,44]],axis=0)
    draws=np.array([diff[rng.integers(0,len(y),len(y))].mean() for _ in range(2000)])
    boot[baseline]={'nll_difference':float(diff.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'scope':'paired test images, conditional on these three head runs and the fixed backbone'}
(R/'bootstrap.json').write_text(json.dumps(boot,indent=2))
selected=json.loads((R/'selected.json').read_text())
fig,ax=plt.subplots(figsize=(7.5,3.5))
for method,label,color in [('adam','Adam',colors[0]),('ivon','IVON BMA (20)',colors[3])]:
    rr=json.loads((R/f'{method}_s42_lr{selected[method]:g}.json').read_text())
    ax.plot([h['epoch'] for h in rr['history']],[h['val_nll'] for h in rr['history']],'-o',ms=3,color=color,label=label)
ax.set(xlabel='Training epoch',ylabel='Validation NLL');ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(F/'validation.pdf');fig.savefig(F/'validation.png');plt.close(fig)
print(json.dumps({'summary':summary,'bootstrap':boot,'risk':risk},indent=2))
