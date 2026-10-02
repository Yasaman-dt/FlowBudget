"""Sequential, resumable full-data cost benchmark. Never reuses untimed samples."""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from flowbudget_config import resource_path, gpu_default, load_config
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name('timed_worker.py')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', default=gpu_default())
    p.add_argument('--estimator-repeats', type=int, default=1)
    p.add_argument('--sweep-repeats', type=int, default=1)
    p.add_argument('--resume-saved-plan', action='store_true',
                   help='Keep saved model/budget settings, changing only repetition counts; archive prior reports.')
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if min(a.estimator_repeats, a.sweep_repeats) < 1:
        p.error('Repeat counts must be positive')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out.parent / f'gpu_{a.gpu}_cost_benchmark.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = load_config(ROOT/'configs/probe_sensitivity.json')['pairs']
    sat = json.loads((ROOT/'empirical_saturation/delta_0p5_fixed_grid/saturation_config.json').read_text())
    config.append(dict(model='SiT-XL/2', solver='euler-maruyama', script='SiT_Imagenet/estimate_em_budget.py',
                       args=['--data-path', str(resource_path('datasets/imagenet_2012/images/train')),'--vae','ema']))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED='1',
               PATH=str(Path(sys.executable).parent)+':'+os.environ.get('PATH',''),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               TF_NUM_INTRAOP_THREADS='4', TF_NUM_INTEROP_THREADS='2',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    # Fixed host thread limits for both methods; no parallel jobs from this runner.
    plan = dict(N=500, tau=.01, images_per_budget=10000, gpu=a.gpu,
                estimator_repeats=a.estimator_repeats, sweep_repeats=a.sweep_repeats,
                pairs=[], scope='Full measured fixed-grid sweep; no interpolation or extrapolated runtimes')
    for i, spec in enumerate(config):
        curve=sat['curves'][i]
        with (ROOT/curve['path']).open() as h:
            budgets=sorted(int(r[curve['k_column']]) for r in csv.DictReader(h)
                           if int(r[curve['k_column']]) in sat['allowed_steps'])
        plan['pairs'].append(dict(**spec, budgets=budgets,
                                 nfe_per_image=[k*(1 if spec['solver']=='euler' else 2) for k in budgets]))
    planpath=out/'plan.json'
    if a.resume_saved_plan:
        if not planpath.exists():
            raise FileNotFoundError('Resuming requires an existing plan.json')
        previous=json.loads(planpath.read_text())
        if previous['gpu'] != a.gpu:
            raise ValueError('Resume on the same GPU as the saved plan')
        plan=dict(previous, estimator_repeats=a.estimator_repeats, sweep_repeats=a.sweep_repeats)
        archive=out/'report_history'/str(time.time_ns())
        archive.mkdir(parents=True)
        for name in ('plan.json','results.csv','results.json','paper_table.md'):
            if (out/name).exists(): shutil.copy2(out/name,archive/name)
        print(f'Archived previous reports in {archive}; retaining saved budgets and all raw timing records.',flush=True)
    if not a.resume_saved_plan and planpath.exists() and json.loads(planpath.read_text()) != plan:
        raise ValueError('Output belongs to another plan')
    planpath.write_text(json.dumps(plan,indent=2)+'\n')
    if a.plan_only:
        print(planpath); return
    inventory = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    (out/'gpu_inventory.txt').write_text(inventory)
    uuid=next(line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip()==a.gpu)
    records=[]
    def idle():
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if any(uuid in line for line in active.splitlines()):
            raise RuntimeError('GPU has another process. Resume this queue when it is idle; no process was killed.')
    def run(folder, kind, script, args, warmup, cwd, fid_gpu=False, ddp=False):
        folder.mkdir(parents=True,exist_ok=True)
        timing=folder/'timing.json'
        if timing.exists():
            return json.loads(timing.read_text())
        if (folder/'started.json').exists():
            raise RuntimeError(f'Incomplete run at {folder}; inspect before rerunning. Cached samples cannot be timed.')
        idle()
        job=dict(kind=kind,script=str(script),args=args,warmup_args=warmup,timing_output=str(timing))
        (folder/'job.json').write_text(json.dumps(job,indent=2))
        command=[sys.executable,str(WORKER),'--job',str(folder/'job.json')]
        if ddp:
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',*command[1:]]
        jobenv=dict(env)
        if kind=='evaluation' and not fid_gpu: jobenv['CUDA_VISIBLE_DEVICES']=''
        (folder/'started.json').write_text(json.dumps(dict(command=command,unix_time=time.time(),cwd=str(cwd))))
        print(f'{folder}: {kind}',flush=True)
        started=time.perf_counter()
        with (folder/'run.log').open('w') as log:
            process=subprocess.Popen(command,cwd=cwd,env=jobenv,stdout=log,stderr=subprocess.STDOUT)
            interference=[]
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    listing=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True)
                    allowed={process.pid}
                    tree=[tuple(map(int,line.split())) for line in listing.splitlines()]
                    for _ in range(20):
                        expanded=allowed | {pid for pid,ppid in tree if ppid in allowed}
                        if expanded==allowed: break
                        allowed=expanded
                    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                    interference.extend(line for line in active.splitlines() if uuid in line and int(line.split(',')[1]) not in allowed)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode,command)
            if interference:
                (folder/'invalid_interference.json').write_text(json.dumps(interference))
                if timing.exists(): timing.rename(folder/'invalid_timing.json')
                raise RuntimeError('Competing GPU process detected; timing excluded.')
        elapsed=time.perf_counter()-started
        r=json.loads(timing.read_text()); r['process_wall_seconds']=elapsed
        timing.write_text(json.dumps(r,indent=2)+'\n')
        return r
    def sample_eval(i, spec, k, folder):
        sit=ROOT/'SiT_Imagenet'; modern=ROOT/'rectified_flow_modern'; rf=ROOT/'rectified_flow_cifar10'
        if i<3:
            script=rf/'sample_rf_cifar10_euler.py'
            common=['--checkpoint',str(resource_path(rf/f'checkpoints/cifar10_{i+1}_rectified_flow.pth')),'--batch-size','128','--seed','0']
            args=common+['--k',str(k),'--num-samples','10000','--output-dir',str(folder/'samples')]
            warm=common+['--k','2','--num-samples','128','--output-dir',str(folder/'warmup')]
            npz=folder/'samples/images.npz'; reference=resource_path(rf/'references/cifar10_test_10k.npz')
        elif i==3:
            script=modern/'sample_unet_euler_sweep.py'
            common=['--batch-size','128','--seed','0','--no-fid']
            args=common+['--steps',str(k),'--num-samples','10000','--output-dir',str(folder/'samples')]
            warm=common+['--steps','2','--num-samples','128','--output-dir',str(folder/'warmup')]
            npz=folder/f'samples/k_{k:03d}/images.npz'; reference=resource_path(modern/'references/cifar10_test_10k.npz')
        else:
            script=sit/'sample_ddp.py'
            mode='SDE' if i==6 else 'ODE'
            common=[mode,'--model','SiT-XL/2','--image-size','256','--vae','mse' if i==5 else 'ema',
                    '--per-proc-batch-size','8','--cfg-scale','1.0','--global-seed','0','--learn-sigma']
            common+=['--sampling-method', 'Euler' if i==6 else ('heun2' if i==5 else 'euler')]
            if i==6: common+=['--diffusion-form','sigma','--diffusion-norm','1.0','--last-step','None']
            args=common+['--num-sampling-steps',str(k+1),'--num-fid-samples','10000','--sample-folder',str(folder/'samples')]
            warm=common+['--num-sampling-steps','3','--num-fid-samples','8','--sample-folder',str(folder/'warmup')]
            npz=folder/'samples.npz'; reference=resource_path(sit/'references/imagenet_train_10k_center_crop_256.npz')
        generation=run(folder/'generation','generation',script,args,warm,script.parent,ddp=i>=4)
        evaluator=resource_path(modern/'evaluations/evaluator.py') if i==3 else resource_path(ROOT.parent/'guided-diffusion/evaluations/evaluator.py')
        evaluation=run(folder/'evaluation','evaluation',evaluator,[str(reference),str(npz)],[],evaluator.parent,fid_gpu=i==6)
        row=dict(pair=i,model=spec['model'],solver=spec['solver'],K=k,
                 nfe_per_image=k*(1 if i<5 else 2),generated_images=10000,
                 generation_seconds=generation['process_wall_seconds'],evaluation_seconds=evaluation['process_wall_seconds'],
                 total_seconds=generation['process_wall_seconds']+evaluation['process_wall_seconds'])
        (folder/'combined.json').write_text(json.dumps(row,indent=2))
        # Remove only this benchmark's newly generated image artifacts after metrics succeed.
        # Timing records and complete evaluator logs remain. Cleanup is outside both timers.
        for name in ('samples','warmup','samples.npz','warmup.npz'):
            artifact=folder/name
            if artifact.is_dir(): shutil.rmtree(artifact)
            elif artifact.is_file(): artifact.unlink()
        return row
    def report():
        (out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
        if records:
            keys=sorted(set().union(*(r.keys() for r in records)))
            with (out/'results.csv').open('w',newline='') as h:
                w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(records)
    for i,spec in enumerate(plan['pairs']):
        pairfolder=out/f'pair_{i}'
        for rep in range(a.estimator_repeats):
            folder=pairfolder/f'estimator_{rep}';folder.mkdir(parents=True,exist_ok=True)
            common=[*spec['args'],'--batch-size','1','--seed','0','--tau','0.01']
            if i==3: common+=['--derivative-method','autograd']
            args=common+['--num-samples','500','--output',str(folder/'estimate.json')]
            warm=common+['--num-samples','2','--output',str(folder/'warmup.json')]
            script=ROOT/spec['script']
            r=run(folder,'estimator',script,args,warm,script.parent)
            records.append(dict(pair=i,model=spec['model'],solver=spec['solver'],phase='estimator',repeat=rep,N=500,
                                estimator_only_seconds=r['estimator_only_seconds'],total_seconds=r['process_wall_seconds'],
                                forward_batch_calls=r['counts']['forward_batch_calls'],jvp_calls=r['counts']['jvp_calls']))
            report()
        pred=json.loads((pairfolder/'estimator_0/estimate.json').read_text())
        khat=pred['budgets'][0]['K_hat'] if i==6 else pred['K_hat']
        for rep in range(a.sweep_repeats):
            for k in spec['budgets']:
                r=sample_eval(i,spec,k,pairfolder/f'sweep_{rep}/k_{k}')
                records.append(dict(**r,phase='sweep_budget',repeat=rep));report()
            r=sample_eval(i,spec,khat,pairfolder/f'selected_{rep}/k_{khat}')
            records.append(dict(**r,phase='selected_budget',repeat=rep));report()
    lines=['Model / Solver | Estimator only (s) | Selection workflow (s) | Full sweep (s) | FlowBudget + selected-budget evaluation (s) | End-to-end speedup',
           '---|---:|---:|---:|---:|---:']
    for i,spec in enumerate(plan['pairs']):
        rr=[r for r in records if r['pair']==i]
        est=[r for r in rr if r['phase']=='estimator']
        selection=statistics.mean(r['total_seconds'] for r in est)
        sweep=statistics.mean(sum(r['total_seconds'] for r in rr if r['phase']=='sweep_budget' and r['repeat']==rep) for rep in range(a.sweep_repeats))
        workflow=selection+statistics.mean(r['total_seconds'] for r in rr if r['phase']=='selected_budget')
        lines.append(f"{spec['model']} / {spec['solver']} | {statistics.mean(r['estimator_only_seconds'] for r in est):.2f} | {selection:.2f} | {sweep:.2f} | {workflow:.2f} | {sweep/workflow:.2f}")
    (out/'paper_table.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
