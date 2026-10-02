"""FlowDCN leading-order EM budget; estimation only, no sampling or FID."""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[2]))
from flowbudget_config import resource_path
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import torch
from run import ROOT, CHECKPOINT, load, save


def drift(velocity, x, t):
    # Linear path score=(t*v-x)/(1-t), w=1-t, f=v+0.5*w*score.
    return (1 + .5*t.reshape(-1,1,1,1))*velocity(x,t)-.5*x


def coefficient(velocity, x, t, directions, increments, epsilon=1e-8):
    """Hutchinson q^2 with matched Rademacher probes and central differences."""
    g=(1-t).sqrt()
    gp=-.5/g
    sums={h: torch.zeros(x.shape[0],device=x.device,dtype=torch.float64) for h in increments}
    for z in directions:
        az_direction=g.reshape(-1,1,1,1)*z
        bz=gp.reshape(-1,1,1,1)*z
        for h in increments:
            az=(drift(velocity,x+h*az_direction,t)-drift(velocity,x-h*az_direction,t))/(2*h)
            az=az.flatten(1).double(); b=bz.flatten(1).double()
            sums[h]+=.5*(az.square().sum(1)+b.square().sum(1)+(az+b).square().sum(1))/len(directions)
    denominator=g.double()*math.sqrt(x[0].numel())+epsilon
    return {h:(q.sqrt()/denominator).tolist() for h,q in sums.items()}


def main(a):
    import fcntl
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader,Subset
    sys.path.insert(0,str(ROOT/'SiT_Imagenet'))
    from estimate_euler_budget_fm_unguided import center_crop_arr
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    hs=[.0005,.001,.002]
    plan=dict(model='FlowDCN-XL-2M',solver='uniform Euler-Maruyama',N=a.probes,tau=a.tau,seeds=[0,1,2],
        cfg_scale=1.0,checkpoint=str(CHECKPOINT),trace_probes=8,finite_difference_increments=hs,
        probe_time_interval=[.01,.99],drift='(1+0.5*t)*v-0.5*x',diffusion='sqrt(1-t)*I',
        formula='ceil(mean(q/(norm(G)+epsilon))/(sqrt(3)*tau))',epsilon=a.epsilon,
        scope='Leading-order manuscript proxy; not SiT augmented RMS estimator. No terminal Euler step correction. No sampling/FID.',
        caveat='Finite-direction square root has Monte Carlo bias; finite differences and nonsmooth kernels limit accuracy.')
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:
        raise RuntimeError('Different experiment settings in output directory')
    save(out/'plan.json',plan)
    model,vae=load()
    transform=transforms.Compose([transforms.Lambda(lambda im:center_crop_arr(im,256)),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    dataset=ImageFolder(resource_path(ROOT.parent/'datasets/imagenet_2012/images/train'),transform=transform)
    summaries=[]
    for seed in [0,1,2]:
        folder=out/f'seed_{seed}'; folder.mkdir(exist_ok=True)
        path=folder/'estimate.json'
        if not path.exists():
            torch.manual_seed(seed)
            indices=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(seed))[:a.probes].tolist()
            loader=DataLoader(Subset(dataset,indices),batch_size=1,num_workers=0)
            generator=torch.Generator(device='cuda').manual_seed(seed+1)
            ratios={h:[] for h in hs}
            with torch.inference_mode():
                for i,(images,labels) in enumerate(loader):
                    data=vae.encode(images.cuda()).latent_dist.sample()*.18215
                    noise=torch.randn_like(data); t=.01+.98*torch.rand(1,device='cuda')
                    x=(1-t.reshape(-1,1,1,1))*noise+t.reshape(-1,1,1,1)*data
                    labels=labels.cuda()
                    z=[torch.randint(0,2,x.shape,device='cuda',generator=generator).to(x.dtype)*2-1 for _ in range(8)]
                    values=coefficient(lambda xx,tt:model(xx,tt,labels),x,t,z,hs,epsilon=a.epsilon)
                    for h in hs:
                        if not all(math.isfinite(v) and v>=0 for v in values[h]): raise RuntimeError('Nonfinite coefficient')
                        ratios[h].extend(values[h])
                    if (i+1)%25==0 or i==0:
                        print(f'Seed {seed}: {i+1}/{a.probes} probes',flush=True)
                        save(folder/'progress.json',dict(completed=i+1,ratios=ratios))
            results={str(h):dict(C_EM=statistics.mean(v),K_hat=max(1,math.ceil(statistics.mean(v)/(math.sqrt(3)*a.tau)))) for h,v in ratios.items()}
            k=results['0.001']['K_hat']; ks=[v['K_hat'] for v in results.values()]
            spread=(max(ks)-min(ks))/k
            save(path,dict(seed=seed,N=a.probes,tau=a.tau,K_hat=k,NFE_hat_unguided_uniform_EM=k,
                 selected_h=.001,results=results,ratios=ratios,dataset_indices=indices,
                 relative_budget_spread=spread,sensitivity_passed=spread<=.05,settings=plan))
        report=json.loads(path.read_text())
        if (report['seed'], report['N'], report['tau']) != (seed, a.probes, a.tau):
            raise RuntimeError('Cached estimate settings differ; use a new output directory')
        summaries.append(dict(seed=seed,N=a.probes,tau=a.tau,K_hat=report['K_hat'],
            sensitivity_passed=report['sensitivity_passed'],relative_budget_spread=report['relative_budget_spread'],source=str(path)))
        with (out/'per_seed.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
        print(f"Seed {seed} finished: K_hat={report['K_hat']}, sensitivity_passed={report['sensitivity_passed']}",flush=True)
    ks=[s['K_hat'] for s in summaries]
    save(out/'summary.json',dict(mean_K_hat=statistics.mean(ks),std_K_hat=statistics.stdev(ks),
        all_sensitivity_passed=all(s['sensitivity_passed'] for s in summaries),per_seed=summaries,settings=plan))
    print('Completed three EM estimation seeds',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,
                   default=Path(__file__).resolve().parents[1] / 'results' / 'estimates' / 'estimation_sde_em')
    p.add_argument('--probes',type=int,default=2000)
    p.add_argument('--tau',type=float,default=0.01,help='Use 0.015 for the final-paper SDE setting')
    p.add_argument('--epsilon',type=float,default=1e-8)
    a=p.parse_args()
    if a.probes<1:p.error('probes must be positive')
    if not math.isfinite(a.tau) or a.tau <= 0:p.error('tau must be finite and positive')
    if not math.isfinite(a.epsilon) or a.epsilon <= 0:p.error('epsilon must be finite and positive')
    main(a)
