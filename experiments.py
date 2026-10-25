"""Six distinct >=30-epoch configurations: baseline plus controlled ablations."""
import subprocess
import sys
import csv
import json
from pathlib import Path

jobs = [('baseline',[])]
jobs += [(f'margin_{m}',['--margin',str(m)]) for m in (0.1,0.5)]
jobs += [(f'dim_{d}',['--dim',str(d)]) for d in (64,256)]
jobs += [('random',['--mining','random'])]
# 0.3/128/hard is reused from baseline, avoiding redundant runs.
# Six distinct jobs cover all requested ablation values.
for name,flags in jobs:
    subprocess.run([sys.executable,'train.py','--epochs','30','--out',f'runs/{name}']+flags,check=True)
rows = []
for name,_ in jobs:
    r = json.loads(Path(f'runs/{name}/results.json').read_text())
    c = r['config']
    rows.append({'run':name,'lr':c['lr'],'batch_size':r['batch_size'],'margin':c['margin'],
                 'dim':c['dim'],'mining':c['mining'],'optimizer':c['optimizer'],
                 'val_triplet_accuracy':r['val_triplet_accuracy'],**r['test']})
with open('runs/results.csv','w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
