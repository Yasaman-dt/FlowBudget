"""Run an existing pipeline script with warm-up and synchronized wall timing.

Only estimator timing boundaries are inserted into an in-memory AST. Original
estimator mathematics and source files are unchanged. No per-call GPU fences.
"""
import argparse
import ast
import gc
import json
import os
from pathlib import Path
import sys
import time

ENTRY = time.perf_counter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    import torch
    if job['kind'] != 'evaluation' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing a CPU timing fallback')
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    measurements = {}
    def mark(name):
        sync()
        measurements[name] = time.perf_counter()
    counts = dict(forward_batch_calls=0, forward_sample_calls=0, jvp_calls=0)
    def hook(module, inputs):
        if type(module).__name__ in ('SongUNet', 'NCSNpp', 'SiT'):
            counts['forward_batch_calls'] += 1
            if inputs and hasattr(inputs[0], 'shape'):
                counts['forward_sample_calls'] += int(inputs[0].shape[0])
    handle = torch.nn.modules.module.register_module_forward_pre_hook(hook)
    original_jvp = torch.autograd.functional.jvp
    def counted_jvp(*args, **kwargs):
        counts['jvp_calls'] += 1
        return original_jvp(*args, **kwargs)
    torch.autograd.functional.jvp = counted_jvp

    def execute(argv, instrument=False):
        script = Path(job['script']).resolve()
        tree = ast.parse(script.read_text(), filename=str(script))
        if instrument:
            # Begin before sample selection/loader construction; end after K aggregation.
            starts = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'indices' for t in n.targets)]
            if len(starts) != 1:
                raise ValueError(f'Expected one probe-selection boundary: {script}')
            end_candidates = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'result' for t in n.targets)]
            if script.name == 'estimate_em_budget.py':
                end_candidates = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                                  and any(isinstance(t, ast.Subscript) and ast.unparse(t) == "result['budgets']" for t in n.targets)]
            if len(end_candidates) != 1:
                raise ValueError(f'Expected one aggregation boundary: {script}')
            class Markers(ast.NodeTransformer):
                def visit_Assign(self, node):
                    if node is starts[0]:
                        return [ast.parse("__benchmark_mark('estimator_start')").body[0], node]
                    if node is end_candidates[0]:
                        return [node, ast.parse("__benchmark_mark('estimator_end')").body[0]]
                    return node
            tree = Markers().visit(tree)
            ast.fix_missing_locations(tree)
        if job['kind'] == 'evaluation':
            # TensorFlow session.run is blocking; each feature/metric phase ends on return.
            for function in tree.body:
                if isinstance(function, ast.FunctionDef) and function.name == 'main':
                    body=[]
                    for node in function.body:
                        text=ast.unparse(node)
                        phase=None
                        for needle,name in [('evaluator.warmup(', 'fid_warmup'),
                                            ('ref_acts =', 'reference_features'),
                                            ('ref_stats,', 'reference_statistics'),
                                            ('sample_acts =', 'sample_features'),
                                            ('sample_stats,', 'sample_statistics'),
                                            ('print(\'FID:\'', 'fid_distance'),
                                            ('print(\'sFID:\'', 'sfid_distance'),
                                            ('evaluator.compute_inception_score(', 'inception_score'),
                                            ('evaluator.compute_prec_recall(', 'precision_recall')]:
                            if needle in text:
                                phase=name;break
                        if phase:
                            body.extend(ast.parse(f"__benchmark_mark('{phase}_start')").body)
                        body.append(node)
                        if phase:
                            body.extend(ast.parse(f"__benchmark_mark('{phase}_end')").body)
                    function.body=body
            ast.fix_missing_locations(tree)
        sys.path.insert(0, str(script.parent))
        sys.argv = [str(script), *argv]
        # sample_ddp changes grad mode; restore it between warm-up and measurement.
        torch.set_grad_enabled(True)
        namespace = dict(__name__='__main__', __file__=str(script), __benchmark_mark=mark)
        exec(compile(tree, str(script), 'exec'), namespace)
        sync()
        sys.path.pop(0)
        del namespace
        gc.collect()

    warmup_start = time.perf_counter()
    if job.get('warmup_args'):
        execute(job['warmup_args'])
    sync()
    warmup_seconds = time.perf_counter() - warmup_start
    counts.update({key: 0 for key in counts})
    mark('work_start')
    execute(job['args'], instrument=job['kind'] == 'estimator')
    mark('work_end')
    handle.remove()
    result = dict(job=job, work_seconds=measurements['work_end']-measurements['work_start'],
                  warmup_seconds=warmup_seconds, worker_total_seconds=time.perf_counter()-ENTRY,
                  phase_seconds={key[:-6]: measurements[key[:-6]+'_end']-value for key,value in measurements.items()
                                 if key.endswith('_start') and key[:-6]+'_end' in measurements},
                  counts=counts, gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                  visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=torch.__version__,
                  estimator_only_seconds=(measurements['estimator_end']-measurements['estimator_start'])
                  if 'estimator_end' in measurements else None,
                  count_note='Model forward calls only; JVP/backward work is included in time, not converted to equivalent NFE.')
    Path(job['timing_output']).write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
