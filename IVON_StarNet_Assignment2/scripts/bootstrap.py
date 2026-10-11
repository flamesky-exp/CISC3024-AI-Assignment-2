from pathlib import Path
import shutil, urllib.request, json, hashlib
ROOT = Path(__file__).resolve().parents[1]
OLD = Path(r'D:\CISC3024 project 1')
for name in ['src', 'vendor/ivon', 'data/splits', 'checkpoints', 'results', 'report/assets', 'output/pdf', 'tmp/pdfs', 'LICENSES', 'ai_logs']:
    (ROOT/name).mkdir(parents=True, exist_ok=True)
for a,b in [('src/models/starnet.py','src/starnet.py'),('data/splits/cifar10_seed42.json','data/splits/cifar10_seed42.json'),('checkpoints/experiment/starnet_s050_best.pt','checkpoints/backbone.pt'),('report/assets/UM_Logo.png','report/assets/UM_Logo.png'),('LICENSES/Apache-2.0.txt','LICENSES/Apache-2.0.txt')]:
    shutil.copy2(OLD/a, ROOT/b)
base='https://raw.githubusercontent.com/team-approx-bayes/ivon/main/'
manifest={}
for remote,local in [('ivon/_ivon.py','vendor/ivon/_ivon.py'),('ivon/__init__.py','vendor/ivon/__init__.py'),('LICENSE','LICENSES/IVON-GPL-3.0.txt'),('README.md','vendor/IVON_README.md')]:
    body=urllib.request.urlopen(base+remote,timeout=45).read()
    (ROOT/local).write_bytes(body)
    manifest[local]={'url':base+remote,'sha256':hashlib.sha256(body).hexdigest()}
manifest['backbone_sha256']=hashlib.sha256((ROOT/'checkpoints/backbone.pt').read_bytes()).hexdigest()
(ROOT/'results/provenance.json').write_text(json.dumps(manifest,indent=2))
import torch
obj=torch.load(ROOT/'checkpoints/backbone.pt',map_location='cpu',weights_only=False)
print('CHECKPOINT KEYS',list(obj))
print('CUDA',torch.cuda.is_available(),torch.cuda.get_device_name(0))
print('Bootstrap complete',ROOT,flush=True)
