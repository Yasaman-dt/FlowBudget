"""Unguided FlowDCN Euler: matched-probe finite differences and FID-10K."""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[2]))
from flowbudget_config import resource_path
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
CHECKPOINT = resource_path(REPO / 'checkpoints/FlowDCN-XL-2M-R256.pth')
REFERENCE = resource_path(ROOT / 'SiT_Imagenet/references/imagenet_train_10k_center_crop_256.npz')
EVALUATOR = resource_path(ROOT.parent / 'guided-diffusion/evaluations/evaluator.py')


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def load():
    import torch
    from diffusers import AutoencoderKL
    sys.path.insert(0, str(REPO))
    from src.models.flowdcn import FlowDCN
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = FlowDCN(input_size=32, patch_size=2, in_channels=4, num_groups=16,
                    hidden_size=1152, num_blocks=28, num_classes=1000, learn_sigma=True)
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    if 'optimizer_states' in state:
        # Official ModelLoader uses the EMA parameter list for Lightning checkpoints.
        ema = state['optimizer_states'][0]['ema']
        params = list(model.named_parameters())
        if len(ema) != len(params):
            raise RuntimeError('EMA parameter count mismatch')
        with torch.no_grad():
            for (name, param), value in zip(params, ema):
                if param.shape != value.shape:
                    raise RuntimeError(f'EMA shape mismatch: {name}')
                param.copy_(value)
        print('Loaded official EMA parameters, count and shapes verified', flush=True)
    else:
        model.load_state_dict(state, strict=True)
    del state
    model = model.cuda().eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema', local_files_only=True)
    vae = vae.cuda().eval().requires_grad_(False)
    print('Checkpoint loaded strictly; parameters:', sum(p.numel() for p in model.parameters()), flush=True)
    return model, vae


def estimate(a):
    import torch
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    sys.path.insert(0, str(ROOT / 'SiT_Imagenet'))
    from estimate_euler_budget_fm_unguided import center_crop_arr
    model, vae = load()
    torch.manual_seed(a.seed)
    transform = transforms.Compose([transforms.Lambda(lambda im: center_crop_arr(im, 256)),
        transforms.ToTensor(), transforms.Normalize([.5]*3, [.5]*3)])
    dataset = ImageFolder(resource_path(ROOT.parent / 'datasets/imagenet_2012/images/train'), transform=transform)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(a.seed))[:a.probes]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=1, num_workers=0)
    hs = [0.0005, 0.001, 0.002]
    ratios = {str(h): [] for h in hs}
    with torch.inference_mode():
        for i, (images, labels) in enumerate(loader):
            data = vae.encode(images.cuda()).latent_dist.sample() * .18215
            noise = torch.randn_like(data)
            t = torch.rand(1, device='cuda') * (1-max(hs))
            x = (1-t[:, None, None, None])*noise + t[:, None, None, None]*data
            labels = labels.cuda()
            v = model(x, t, labels)
            for h in hs:
                acc = (model(x+h*v, t+h, labels)-v)/h
                ratio = (acc.flatten(1).norm(dim=1)/(v.flatten(1).norm(dim=1)+1e-8)).item()
                if not math.isfinite(ratio):
                    raise RuntimeError('Nonfinite finite-difference ratio')
                ratios[str(h)].append(ratio)
            if (i+1) % 25 == 0 or i == 0:
                print(f'Probes {i+1}/{a.probes}: ' + str({h: math.ceil(sum(vv)/len(vv)/(2*a.tau)) for h,vv in ratios.items()}), flush=True)
                save(a.output/'probe_progress.json', dict(completed=i+1, ratios=ratios))
    results = {h: dict(R_theta=sum(vals)/len(vals), K_hat=max(1,math.ceil(sum(vals)/len(vals)/(2*a.tau)))) for h,vals in ratios.items()}
    ks = [r['K_hat'] for r in results.values()]
    spread = (max(ks)-min(ks))/max(1,results['0.001']['K_hat'])
    report = dict(model='FlowDCN-XL-2M', solver='uniform ODE Euler', cfg_scale=1.0,
        N=a.probes, tau=a.tau, seed=a.seed, epsilon=1e-8, time_interval=[0,1-max(hs)],
        derivative='forward finite difference along (v,1)', results=results, ratios=ratios,
        dataset_indices=indices.tolist(), selected_h=.001, K_hat=results['0.001']['K_hat'],
        relative_budget_spread=spread, sensitivity_passed=spread <= .05)
    save(a.output/'estimate.json', report)
    if spread > .05:
        raise RuntimeError(f'Finite-difference sensitivity failed: budget spread {spread:.2%}; sweep not started')


def sample(a):
    import numpy as np
    from PIL import Image
    import torch
    from torchvision.utils import save_image
    model, vae = load()
    torch.manual_seed(a.seed)
    folder = a.output/f'k_{a.k:04d}'
    brownian = torch.Generator(device='cuda').manual_seed(a.seed + 100000)
    folder.mkdir(exist_ok=True)
    image_dir = folder/'images'
    image_dir.mkdir(exist_ok=True)
    path = folder/'images.npy'
    pixels = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=(a.images,256,256,3))
    with torch.inference_mode():
        for offset in range(0,a.images,a.batch_size):
            n = min(a.batch_size,a.images-offset)
            x = torch.randn(n,4,32,32,device='cuda')
            labels = torch.randint(0,1000,(n,),device='cuda')
            for step in range(a.k):
                t = step/a.k
                v = model(x, torch.full((n,),t,device='cuda'),labels)
                if a.solver == 'em':
                    # Uniform native-preserve SDE, without a special final ODE step.
                    f = (1+.5*t)*v-.5*x
                    z = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=brownian)
                    x = x + f/a.k + math.sqrt((1-t)/a.k)*z
                elif a.solver == 'heun':
                    predicted = x + v/a.k
                    v_next = model(predicted, torch.full((n,), (step+1)/a.k, device='cuda'), labels)
                    x = x + (v + v_next)/(2*a.k)
                else:
                    x = x + v/a.k
            decoded = vae.decode(x/.18215).sample
            if not torch.isfinite(decoded).all():
                raise RuntimeError('Nonfinite sample')
            if offset == 0:
                save_image(decoded,folder/'image_grid.png',nrow=4,normalize=True,value_range=(-1,1))
            pixels[offset:offset+n] = torch.clamp(127.5*decoded+128,0,255).permute(0,2,3,1).to(torch.uint8).cpu().numpy()
            if not a.skip_png:
                for index in range(offset, offset+n):
                    Image.fromarray(pixels[index]).save(image_dir/f'{index:06d}.png')
            if offset % (a.batch_size*25) == 0:
                print(f'K={a.k}: {offset+n}/{a.images} images',flush=True)
    pixels.flush()
    np.savez(folder/'images.npz', pixels)
    del pixels
    path.unlink()
    save(folder/'sampling.json',dict(K=a.k,NFE=2*a.k if a.solver == 'heun' else a.k,
        images=a.images,seed=a.seed,cfg_scale=1.0,batch_size=a.batch_size,solver=a.solver))


def run(a):
    import fcntl
    lock = (a.output/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not REFERENCE.is_file() or not CHECKPOINT.is_file():
        raise FileNotFoundError('Checkpoint or ImageNet reference missing')
    plan = dict(model='FlowDCN-XL-2M',gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), tau=a.tau,
        probes=a.probes,images=a.images,seed=a.seed,batch_size=a.batch_size,cfg_scale=1.0,
        checkpoint=str(CHECKPOINT),reference=str(REFERENCE),finite_difference_steps=[.0005,.001,.002],
        sensitivity_threshold=.05,sweep_multipliers=[.5,.75,1,1.25,1.5],additional_K=[250])
    if (a.output/'plan.json').exists() and json.loads((a.output/'plan.json').read_text()) != plan:
        raise RuntimeError('Output directory has different settings')
    save(a.output/'plan.json',plan)
    def stage(mode,log,k=None):
        command=[sys.executable,str(Path(__file__).resolve()),mode,'--output',str(a.output),
            '--probes',str(a.probes),'--images',str(a.images),'--seed',str(a.seed),
            '--tau',str(a.tau),'--batch-size',str(a.batch_size)]
        if k is not None: command += ['--k',str(k)]
        print('Starting',mode,k,flush=True)
        with log.open('w') as handle:
            subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
    if not (a.output/'estimate.json').exists(): stage('estimate',a.output/'estimator.log')
    estimate_result=json.loads((a.output/'estimate.json').read_text())
    if not estimate_result['sensitivity_passed']: raise RuntimeError('Sensitivity check failed')
    k=estimate_result['K_hat']
    budgets=sorted({max(1,math.ceil(k*f)) for f in [.5,.75,1,1.25,1.5]}|{250})
    save(a.output/'sweep_plan.json',dict(K_hat=k,budgets=budgets,images_per_budget=a.images))
    rows=[]
    for budget in budgets:
        folder=a.output/f'k_{budget:04d}'
        folder.mkdir(exist_ok=True)
        metrics=folder/'metrics.json'
        if not metrics.exists():
            if not (folder/'sampling.json').exists(): stage('sample',folder/'sampling.log',budget)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='2')
            print('Scoring FID for K=',budget,flush=True)
            with (folder/'fid.log').open('w') as handle:
                subprocess.run([sys.executable,str(EVALUATOR),str(REFERENCE),str(folder/'images.npz')],
                    cwd=EVALUATOR.parent,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
            match=re.search(r'\bFID\s*:\s*([0-9.eE+-]+)',(folder/'fid.log').read_text())
            if not match: raise RuntimeError('FID missing from evaluator output')
            save(metrics,dict(k_euler_updates=budget,nfe=budget,fid=float(match[1]),num_samples=a.images,cfg_scale=1.0))
        rows.append(json.loads(metrics.read_text()))
        with (a.output/'fid_vs_k.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save(a.output/'complete.json',dict(status='complete',budgets=budgets))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run','estimate','sample'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probes',type=int,default=2000)
    p.add_argument('--tau',type=float,default=.01)
    p.add_argument('--images',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--k',type=int,default=2)
    p.add_argument('--solver',choices=['euler','em','heun'],default='euler')
    p.add_argument('--skip-png',action='store_true',help='Save the FID NPZ without individual PNG files')
    a=p.parse_args(); a.output=a.output.resolve(); a.output.mkdir(parents=True,exist_ok=True)
    if min(a.probes,a.images,a.batch_size,a.k)<=0 or a.tau<=0: p.error('Counts and tau must be positive')
    globals()[a.mode](a)
