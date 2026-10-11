from pathlib import Path
import subprocess, json, re
from pypdf import PdfReader
from PIL import Image, ImageOps, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
pdf=ROOT/'output/pdf/IVON_StarNet_CIFAR10_Report.pdf'
reader=PdfReader(pdf)
pages=[p.extract_text() for p in reader.pages]
for i,t in enumerate(pages,1):print('PAGE',i,'chars',len(t),'start',repr(t[:90]),'end',repr(t[-90:]))
assert all('??' not in t for t in pages),'Unresolved reference'
log=(ROOT/'output/pdf/main.log').read_text(errors='replace')
print('TEX DIAGNOSTICS', '\n'.join(x for x in log.splitlines() if any(k in x.lower() for k in ['overfull','underfull','warning','undefined'])))
dest=ROOT/'tmp/pdfs';dest.mkdir(parents=True,exist_ok=True)
poppler=Path(r'C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe')
subprocess.run([str(poppler),'-r','100','-png',str(pdf),str(dest/'report')],check=True,capture_output=True)
imgs=sorted(dest.glob('report-*.png'))[:len(pages)]
for group in range(0,len(imgs),6):
    subset=imgs[group:group+6];sheet=Image.new('RGB',(1050,1510),'#dddddd');draw=ImageDraw.Draw(sheet)
    for j,p in enumerate(subset):
        im=Image.open(p).convert('RGB');im.thumbnail((500,465))
        x=(j%2)*525+(525-im.width)//2;y=(j//2)*503+25
        sheet.paste(im,(x,y));draw.text(((j%2)*525+12,(j//2)*503+7),f'Page {group+j+1}',fill='black')
    sheet.save(dest/f'contact-{group//6+1}.jpg')
(ROOT/'results/pdf_qa.json').write_text(json.dumps({'page_count':len(pages),'unresolved_references':False,'rendered_pages':len(imgs),'latex_warnings':[x for x in log.splitlines() if 'Warning' in x or 'Overfull' in x]},indent=2))
print('PAGE COUNT',len(pages))
print('Original backbone reference',json.loads((ROOT/'results/original_backbone_reference.json').read_text())['accuracy'])
