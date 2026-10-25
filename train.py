"""Single entry point: train, select by validation, export and evaluate test."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import EmbeddingModel
from data import load, PKSampler, seed_all
from loss import triplet_loss
from optimizers import Optimizer
from eval import embed, evaluate, triplet_accuracy

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',default='data')
    p.add_argument('--out',default='runs/baseline')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--dim',type=int,default=128)
    p.add_argument('--margin',type=float,default=0.3)
    p.add_argument('--mining',choices=['hard','random'],default='hard')
    p.add_argument('--optimizer',choices=['adam','nadam','sgd'],default='adam')
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--beta1',type=float,default=0.9)
    p.add_argument('--beta2',type=float,default=0.999)
    p.add_argument('--eps',type=float,default=1e-8)
    p.add_argument('--momentum',type=float,default=0.9)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--p',type=int,default=16)
    p.add_argument('--k',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable. Use the CUDA-enabled .venv Python.')
    print(f'Device: {args.device}; torch: {torch.__version__}; CUDA runtime: {torch.version.cuda}', flush=True)
    if args.device.startswith('cuda'):
        print(f'GPU: {torch.cuda.get_device_name(torch.device(args.device))}', flush=True)
    if args.epochs < 1:
        p.error('epochs must be positive; final submission requires >=30')
    seed_all(args.seed)
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    (train,val,test),classes = load(args.root,args.seed)
    np.savez(out/'split_ids.npz',train=train.ids,val=val.ids,test=test.ids)
    sampler = PKSampler(train.targets,args.p,args.k,args.seed)
    loader = DataLoader(train,batch_sampler=sampler,num_workers=0)
    model = EmbeddingModel(args.dim).to(args.device)
    opt = Optimizer(model.parameters(),args.optimizer,args.lr,args.beta1,args.beta2,
                    args.eps,args.momentum,args.weight_decay)
    best = -1.0
    with (out/'history.jsonl').open('w') as history:
        for epoch in range(args.epochs):
            sampler.epoch = epoch
            opt.lr = args.lr*0.5*(1+math.cos(math.pi*epoch/args.epochs))
            model.train(); total = 0.0
            for x,y in loader:
                opt.zero_grad()
                loss = triplet_loss(model(x.to(args.device)),y.to(args.device),args.margin,args.mining)
                loss.backward(); opt.step()
                total += loss.item()
            vz,vy = embed(model,val,args.device)
            vm = evaluate(vz,vy)
            acc = triplet_accuracy(vz,vy,args.seed)
            row = {'epoch':epoch+1,'loss':total/len(loader),'lr':opt.lr,
                   'val_triplet_accuracy':acc,**vm}
            print(json.dumps(row),flush=True)
            history.write(json.dumps(row)+'\n'); history.flush()
            if vm['Recall@1'] > best:
                best = vm['Recall@1']
                torch.save({'model':model.state_dict(),'dim':args.dim,'epoch':epoch+1,
                            'classes':classes},out/'best.pt')
    checkpoint = torch.load(out/'best.pt',map_location=args.device,weights_only=True)
    model.load_state_dict(checkpoint['model'])
    z,y = embed(model,test,args.device)
    np.savez(out/'gallery.npz',embeddings=z,labels=y,images=test.images[test.ids],classes=np.array(classes))
    vz,vy = embed(model,val,args.device)
    result = {'config':vars(args),'batch_size':args.p*args.k,
              'best_epoch':checkpoint['epoch'],'val_triplet_accuracy':triplet_accuracy(vz,vy,args.seed),
              'test':evaluate(z,y)}
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
