import argparse, copy, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from benchmarks.pamap2.models import MCNN, SensorAttentionHAR, TinyHAR

class DeepConvLSTM(nn.Module):
    def __init__(self, d=52, classes=12):
        super().__init__()
        layers=[]; c=1
        for _ in range(4): layers += [nn.Conv2d(c,64,(5,1)),nn.ReLU()]; c=64
        self.conv=nn.Sequential(*layers); self.rnn=nn.LSTM(64*d,128,2,batch_first=True,dropout=.5); self.drop=nn.Dropout(.5); self.fc=nn.Linear(128,classes)
    def encode(self,x):
        z=self.conv(x[:,None]).permute(0,2,1,3).flatten(2); z,_=self.rnn(z); return z
    def encode_tokens(self,x): return self.encode(x)
    def forward(self,x): return self.fc(self.drop(self.encode(x)[:,-1]))

class SpectralBranch(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(6,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(.2),nn.Conv1d(64,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(.2),nn.Conv1d(64,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU())
    def forward(self,x): return self.net(x)

class DeepSense(nn.Module):
    def __init__(self, classes=12):
        super().__init__()
        # PAMAP2 canonical feature layout: heart rate followed by three
        # 17-channel IMUs (temperature, two accelerometers, gyro, magnetometer,
        # orientation). Use every valid triaxial sensor as a DeepSense branch.
        self.sensor_slices=tuple(
            slice(start+offset,start+offset+3)
            for start in (1,18,35) for offset in (1,4,7,10)
        )
        self.sensor_branches=nn.ModuleList([SpectralBranch() for _ in self.sensor_slices])
        self.fuse=nn.Sequential(nn.Conv1d(64*len(self.sensor_slices),64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(.2),nn.Conv1d(64,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(.2),nn.Conv1d(64,120,3,padding=1),nn.BatchNorm1d(120),nn.ReLU())
        self.rnn=nn.GRU(120,120,2,batch_first=True,dropout=.5); self.fc=nn.Linear(120,classes)
    @staticmethod
    def spectrum(x):
        # Preserve frequency bins as a sequence for the fusion CNN and GRU.
        z=torch.fft.rfft(x.float(),dim=1)[:,1:11]
        return torch.cat([z.real.transpose(1,2),z.imag.transpose(1,2)],dim=1)
    def encode(self,x):
        parts=[branch(self.spectrum(x[:,:,sensor_slice])) for branch,sensor_slice in zip(self.sensor_branches,self.sensor_slices)]
        z=self.fuse(torch.cat(parts,1)).transpose(1,2); z,_=self.rnn(z); return z
    def encode_tokens(self,x): return self.encode(x)
    def forward(self,x): return self.fc(self.encode(x)[:,-1])

class SensorAttention(nn.Module):
    def __init__(self,c=64):
        super().__init__(); self.q=nn.Conv1d(c,c,1,bias=False); self.k=nn.Conv1d(c,c,1,bias=False); self.v=nn.Conv1d(c,c,1,bias=False)
        for conv in (self.q, self.k, self.v): nn.init.kaiming_normal_(conv.weight)
        self.gamma=nn.Parameter(torch.tensor([0.]))
    def forward(self,x):
        shape=x.shape; z=x.flatten(2); beta=torch.softmax(torch.bmm(self.q(z).transpose(1,2),self.k(z)),dim=1); return (self.gamma*torch.bmm(self.v(z),beta)+z).reshape(shape)

class AttendDiscriminate(nn.Module):
    def __init__(self,d=52,classes=12):
        super().__init__()
        # Attend-and-Discriminate author architecture (PAMAP2 settings).
        # Four valid temporal convolutions reduce the 24-sample window to 8.
        self.conv1 = nn.Conv2d(1, 64, (5, 1))
        self.conv2 = nn.Conv2d(64, 64, (5, 1))
        self.conv3 = nn.Conv2d(64, 64, (5, 1))
        self.conv4 = nn.Conv2d(64, 64, (5, 1))
        self.sensor = SensorAttention(64)
        self.drop = nn.Dropout(.9)
        self.rnn = nn.GRU(64 * d, 128, 2, bidirectional=False, dropout=0)
        self.temporal_fc = nn.Linear(128, 1)
        self.clsdrop = nn.Dropout(.5)
        self.fc = nn.Linear(128, classes)
        self.register_buffer('centers', torch.randn(classes, 128))
    def encode_tokens(self,x):
        x = x.unsqueeze(1)
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            x = F.relu(conv(x))
        refined = torch.cat(
            [self.sensor(x[:, :, t, :].unsqueeze(3)) for t in range(x.shape[2])],
            dim=-1,
        )
        # Author implementation uses sequence-first GRU and temporal attention.
        sequence = refined.permute(3, 0, 1, 2).reshape(refined.shape[3], refined.shape[0], -1)
        sequence = self.drop(sequence)
        outputs, _ = self.rnn(sequence)
        attention = torch.softmax(self.temporal_fc(outputs).squeeze(2), dim=0).unsqueeze(2)
        return outputs.transpose(0, 1)
    def encode(self,x):
        outputs = self.encode_tokens(x).transpose(0, 1)
        attention = torch.softmax(self.temporal_fc(outputs).squeeze(2), dim=0).unsqueeze(2)
        feature = torch.sum(attention * outputs, dim=0)
        return feature.unsqueeze(1)
    def forward(self,x):
        feature=self.encode(x).squeeze(1)
        z=F.normalize(feature,p=2,dim=1)
        return z,self.fc(self.clsdrop(feature))

MODELS={'deepconvlstm':DeepConvLSTM,'deepsense':DeepSense,'attend_discriminate':AttendDiscriminate,
        'tinyhar':TinyHAR,'mcnn':MCNN,'sahar':SensorAttentionHAR}
PROTOCOLS={
 'deepconvlstm':{'optimizer':'RMSprop','lr':1e-3,'loss':'cross_entropy','dropout':.5},
 'deepsense':{'optimizer':'Adam','lr':1e-4,'betas':[.5,.9],'loss':'cross_entropy+tensorflow_l2_loss','l2':5e-4,'batch_size':64,'internal':'12 PAMAP2 triaxial branches (3 positions x acc16/acc6/gyro/magnetometer) + complex FFT + sensor CNN + fusion CNN + BN/dropout + 2-layer GRU','adaptation':'canonical 24x52 input; heart rate, temperature and invalid orientation excluded from spectral branches'},
 'attend_discriminate':{'optimizer':'Adam','lr':1e-3,'loss':'mixup_cross_entropy+center_loss','mixup_alpha':.8,'center_beta':.003,'center_lr':.001,'scheduler':'StepLR(10,0.9)','dropout':.9,'classifier_dropout':.5,'architecture':'Attend-and-Discriminate author PAMAP2 configuration','author_epochs':300,'author_patience':0,'clip_grad':0},
 'tinyhar':{'optimizer':'Adam','lr':1e-3,'loss':'cross_entropy','architecture':'TinyHAR attn/filter/GRU/FC','adaptation':'same-padded temporal convolutions for canonical 24-sample input'},
 'mcnn':{'optimizer':'Adam','lr':1e-3,'loss':'cross_entropy','architecture':'2 convolution blocks + sensor/temporal fusion','adaptation':'same-padded temporal convolutions for canonical 24-sample input'},
 'sahar':{'optimizer':'Adam','lr':1e-3,'loss':'cross_entropy','architecture':'original SA-HAR sensor attention + 2 custom encoder layers + temporal attention','filters':128,'encoder_dropout':.1,'classifier_dropout':.2,'scheduler':'ReduceLROnPlateau(factor=0.1,patience=5)','author_epochs':150,'author_patience':15,'adaptation':'canonical 24x52 input only'},
}
def metrics(y,p):
    return {'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0),'kappa':cohen_kappa_score(y,p),'recall_per_class':recall_score(y,p,labels=range(12),average=None,zero_division=0).tolist(),'f1_per_class':f1_score(y,p,labels=range(12),average=None,zero_division=0).tolist(),'confusion':confusion_matrix(y,p,labels=range(12)).tolist(),'pred_dist':np.bincount(p,minlength=12).tolist()}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=MODELS,required=True); p.add_argument('--data',default='data/canonical'); p.add_argument('--output',required=True); p.add_argument('--epochs',type=int,default=50); p.add_argument('--patience',type=int,default=7); p.add_argument('--batch_size',type=int,default=256); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--workers',type=int,default=4); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); root=Path(a.data); manifest=json.loads((root/'manifest.json').read_text())
    if not manifest.get('checksum'): raise ValueError('canonical checksum missing')
    arrays={}
    for split in ('train','val','test'):
        parts=[np.load(root/f'subject{s}.npz') for s in manifest['split'][split]]; arrays[split]=(np.concatenate([q['x'] for q in parts]),np.concatenate([q['y'] for q in parts]))
    mean=arrays['train'][0].mean((0,1),keepdims=True); std=arrays['train'][0].std((0,1),keepdims=True)+1e-6
    loaders={}; counts={}
    for split,(x,y) in arrays.items():
        x=(x-mean)/std; counts[split]=np.bincount(y,minlength=12).tolist(); ds=TensorDataset(torch.from_numpy(x.astype(np.float32)),torch.from_numpy(y)); loaders[split]=DataLoader(ds,a.batch_size,shuffle=split=='train',num_workers=a.workers,pin_memory=True)
    print("=== PAMAP2 CANONICAL BENCHMARK ===", flush=True)
    print(f"model={a.model}", flush=True)
    print(f"split=train:{manifest['split']['train']} val:{manifest['split']['val']} test:{manifest['split']['test']}", flush=True)
    print(f"input=window:{manifest['window']} stride:{manifest['stride']} features:52 classes:12", flush=True)
    print(f"checksum={manifest['checksum']}", flush=True)
    for split in ('train','val','test'):
        print(f"samples.{split}={len(arrays[split][1])} labels={counts[split]}", flush=True)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=MODELS[a.model]().to(dev); protocol=PROTOCOLS[a.model]
    if a.model=='deepconvlstm': opt=torch.optim.RMSprop(model.parameters(),lr=1e-3)
    elif a.model=='deepsense': opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.5,.9))
    elif a.model=='attend_discriminate': opt=torch.optim.Adam((p for n,p in model.named_parameters() if n!='centers'),lr=1e-3)
    else: opt=torch.optim.Adam(model.parameters(),lr=protocol['lr'])
    if a.model=='attend_discriminate':
        scheduler=torch.optim.lr_scheduler.StepLR(opt,10,.9)
    elif a.model=='sahar':
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.1,patience=5)
    else:
        scheduler=None
    lossfn=nn.CrossEntropyLoss(); best=-1; state=None; stale=0
    def run(loader,train=False):
        model.train(train); ys=[]; ps=[]; losses=[]
        for x,y in loader:
            x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True)
            if train and a.model=='attend_discriminate':
                # Exact author training order: MixUp CE plus center loss on
                # the normalized representation from the unmixed labels.
                lam=np.random.beta(.8,.8,len(x)); lam=np.maximum(lam,1.0-lam)
                lam=torch.as_tensor(lam,device=dev,dtype=x.dtype)
                order=torch.randperm(len(x),device=dev)
                mixed=x*lam[:,None,None]+x[order]*(1-lam[:,None,None])
                features,logits=model(mixed)
                ce_a=F.cross_entropy(logits,y,reduction='none')
                ce_b=F.cross_entropy(logits,y[order],reduction='none')
                loss=(lam*ce_a+(1-lam)*ce_b).mean()+.003*F.mse_loss(features,model.centers[y])
            else:
                output=model(x)
                if a.model=='attend_discriminate':
                    features,logits=output
                else:
                    logits=output[0] if isinstance(output,tuple) else output
                loss=lossfn(logits,y)
                if train and a.model=='deepsense':
                    # DeepSense uses an explicit L2 term in its objective.
                    # TensorFlow l2_loss is sum(t ** 2) / 2.
                    l2=.5*sum((param * param).sum() for param in model.parameters())
                    loss=loss + protocol['l2'] * l2
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if a.model!='attend_discriminate':
                    nn.utils.clip_grad_norm_(model.parameters(),5)
                opt.step()
            if train and a.model=='attend_discriminate':
                with torch.no_grad():
                    # Equivalent to the repository's get_center_delta().
                    sorted_y, indices=torch.sort(y)
                    target_centers=model.centers[sorted_y]
                    sorted_features=features[indices]
                    delta=target_centers-sorted_features
                    unique, inverse=torch.unique(sorted_y,sorted=True,return_inverse=True)
                    summed=torch.zeros((len(unique),features.size(1)),device=dev,dtype=features.dtype)
                    summed.index_add_(0,inverse,delta)
                    counts=torch.bincount(inverse,minlength=len(unique)).to(features.dtype).unsqueeze(1)
                    center_delta=summed/(counts+1.0)*.001
                    updated=torch.zeros_like(model.centers)
                    updated[unique]=center_delta
                    model.centers.copy_(model.centers-updated)
            losses.append(loss.item()); ys+=y.detach().cpu().tolist(); ps+=logits.argmax(1).detach().cpu().tolist()
        z=metrics(ys,ps); z['loss']=float(np.mean(losses)); return z
    for ep in range(1,a.epochs+1):
        tr=run(loaders['train'],True); va=run(loaders['val'])
        print(
            f"EPOCH {ep:02d}/{a.epochs} | "
            f"TRAIN loss={tr['loss']:.5f} acc={tr['accuracy']:.5f} "
            f"bacc={tr['balanced_accuracy']:.5f} f1={tr['macro_f1']:.5f} kappa={tr['kappa']:.5f} | "
            f"VAL loss={va['loss']:.5f} acc={va['accuracy']:.5f} "
            f"bacc={va['balanced_accuracy']:.5f} f1={va['macro_f1']:.5f} kappa={va['kappa']:.5f}",
            flush=True,
        )
        if va['macro_f1']>best: best=va['macro_f1']; state=copy.deepcopy(model.state_dict()); stale=0
        else: stale+=1
        if scheduler:
            scheduler.step(va['loss']) if a.model=='sahar' else scheduler.step()
        if a.patience>0 and stale>=a.patience: print(f"EARLY_STOPPING epoch={ep} patience={a.patience}",flush=True); break
    model.load_state_dict(state); test=run(loaders['test'])
    result={'model':a.model,'canonical_checksum':manifest['checksum'],'split':manifest['split'],'method_protocol':protocol,'best_val_macro_f1':best,'test':test,'args':vars(a)}
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); torch.save(state,out/'best.pt'); (out/'metrics.json').write_text(json.dumps(result,indent=2))
    print("=== FINAL TEST ===", flush=True)
    print(f"model={a.model} best_val_macro_f1={best:.5f}", flush=True)
    print(f"TEST loss={test['loss']:.5f} acc={test['accuracy']:.5f} bacc={test['balanced_accuracy']:.5f} f1={test['macro_f1']:.5f} kappa={test['kappa']:.5f}", flush=True)
    print(f"saved_metrics={out/'metrics.json'}",flush=True)
if __name__=='__main__': main()
