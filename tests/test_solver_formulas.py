"""Analytic checks of the actual estimator functions and transport solvers."""
import ast
import math
from pathlib import Path
import unittest
import torch
from torch.autograd.functional import jvp
from transport.integrators import ode, sde

ROOT=Path(__file__).resolve().parents[1]


def function(path, name):
    # Load the production function without importing checkpoints/custom GPU kernels.
    tree=ast.parse((ROOT/path).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    ns={'torch':torch,'jvp':jvp,'math':math}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns[name]


class SolverFormulaTests(unittest.TestCase):
    def test_heun_nonlinear_time_dependent_coefficient(self):
        x=torch.tensor([[.2,.4]],dtype=torch.float64)
        t=torch.tensor([.3],dtype=torch.float64)
        velocity=lambda x,t:x.square()+t[:,None]
        v=velocity(x,t); a=2*x*v+1; g=2*x*a; c=2*v.square()
        derivatives=function('SiT_Imagenet/estimate_heun_budget_fm_unguided.py','heun_derivatives')
        for actual,expected in zip(derivatives(velocity,x,t),(v,a,g,c)):
            torch.testing.assert_close(actual,expected)
        fd=function('FlowDCN/flowdcn_experiment/estimate_heun_three_seeds.py','heun_error_vector')
        _,error=fd(velocity,x,t,1e-3)
        torch.testing.assert_close(error,2*g-c,rtol=1e-6,atol=1e-7)

    def test_heun_defect_sign_and_order(self):
        # x'=x^2 has exact update x/(1-h*x); verify both time directions.
        x=.4
        for h in [.01,-.01]:
            exact=x/(1-h*x)
            heun=x+h*(x*x+(x+h*x*x)**2)/2
            leading=h**3*(6*x**4)/12
            self.assertLess(abs((exact-heun)/leading-1),.01)

    def test_ode_step_count_and_solution(self):
        for method,calls_per_step in [('euler',1),('heun2',2)]:
            calls=[]
            def drift(x,t,model):calls.append(t.clone());return x
            k=8
            solver=ode(drift,t0=0,t1=1,sampler_type=method,num_steps=k+1,atol=1e-8,rtol=1e-8)
            result=solver.sample(torch.ones(1,1,dtype=torch.float64),None)
            self.assertEqual(len(calls),calls_per_step*k)
            factor=1+1/k+(0.5/k**2 if method=='heun2' else 0)
            self.assertAlmostEqual(result[-1].item(),factor**k,places=6)

    def test_em_steps_and_noise_normalization(self):
        from unittest.mock import patch
        calls=[]
        def drift(x,t,model):calls.append(t.clone());return torch.ones_like(x)*2
        k=4
        solver=sde(drift,lambda x,t:torch.ones_like(x)*.5,
                   t0=0,t1=1,num_steps=k+1,sampler_type='Euler')
        with patch('torch.randn',side_effect=lambda shape:torch.ones(shape)):
            result=solver.sample(torch.zeros(1,1),None)
        self.assertEqual(len(calls),k)
        # dt*f + sqrt(2*d*dt)*z, with d=0.5, f=2, z=1.
        self.assertAlmostEqual(result[-1].item(),4.)
