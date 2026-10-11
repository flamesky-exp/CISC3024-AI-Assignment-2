from pathlib import Path
import json, zipfile, importlib.metadata as md, hashlib, shutil
ROOT=Path(__file__).resolve().parents[1]
versions={x:md.version(x) for x in ['torch','torchvision','numpy','scipy','matplotlib','Pillow']}
(ROOT/'results/package_versions.json').write_text(json.dumps(versions,indent=2))
shutil.copy2(ROOT/'LICENSES/IVON-GPL-3.0.txt',ROOT/'LICENSE')
files=[]
for folder in ['src','vendor','scripts','report','results','data/splits','LICENSES','ai_logs']:
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc'] and p.name!='delivery_manifest.json':
            files.append(p)
for name in ['README.md','requirements.txt','THIRD_PARTY_NOTICES.md','LICENSE','.gitignore','AIassign2.pdf','output/pdf/IVON_StarNet_CIFAR10_Report.pdf','checkpoints/backbone.pt']:
    files.append(ROOT/name)
selected=json.loads((ROOT/'results/selected.json').read_text())
for method,lr in selected.items():
    for s in [42,43,44]:files.append(ROOT/f'checkpoints/{method}_s{s}_lr{lr:g}.pt')
manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(ROOT/'results/delivery_manifest.json').write_text(json.dumps(manifest,indent=2))
out=ROOT/'output/IVON_StarNet_Assignment2_Source.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,'IVON_StarNet_Assignment2/'+str(p.relative_to(ROOT)))
    z.write(ROOT/'results/delivery_manifest.json','IVON_StarNet_Assignment2/results/delivery_manifest.json')
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(set(z.namelist()))
    for path,digest in manifest.items():
        assert hashlib.sha256(z.read('IVON_StarNet_Assignment2/'+path)).hexdigest()==digest,path
print('Packaged',len(files),'files;',out.stat().st_size,'bytes;',out)
print('Archive integrity, unique paths and SHA-256 manifest verified.')
