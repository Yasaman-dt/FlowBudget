import unittest
import torch
from estimate_em_budget import em_statistics, make_drift, estimate_budget, relative_error

class EMTests(unittest.TestCase):
    def test_nonlinear_derivatives(self):
        x=torch.tensor([[.2,-.4],[.3,.5]],dtype=torch.float64)
        s=torch.tensor([.2,.7],dtype=torch.float64)
        rows=em_statistics(lambda x,s:x.square()+s[:,None]*x,x,s,trace_probes=2,augmented=True)
        for i,r in enumerate(rows):
            g=(2*(1-s[i])).sqrt(); gp=-1/g
            f=x[i].square()+s[i]*x[i]; j=2*x[i]+s[i]
            d=x[i]+j*f+g.square()
            q2=((g*j).square()+gp.square()+g*j*gp).sum()
            self.assertAlmostEqual(r['q2'],q2.item(),places=10)
            self.assertAlmostEqual(r['d2'],d.square().sum().item(),places=10)
            self.assertAlmostEqual(r['g2'],(2*g.square()).item(),places=10)

    def test_fused_sampler_drift(self):
        x=torch.tensor([[.2,-.4]],dtype=torch.float64); s=torch.tensor([.6],dtype=torch.float64)
        velocity=lambda x,s:x.square()+s[:,None]
        v=velocity(x,s); score=(s[:,None]*v-x)/(1-s[:,None])
        torch.testing.assert_close(make_drift(velocity,1.3)(x,s),v+1.3*(1-s[:,None])*score)

    def test_ode_limit(self):
        x=torch.tensor([[.3,.8]],dtype=torch.float64); s=torch.tensor([.4],dtype=torch.float64)
        rows=em_statistics(lambda x,s:2*x,x,s,norm=0,trace_probes=1)
        self.assertEqual(rows[0]['q2'],0)
        with self.assertRaises(ValueError):
            relative_error(rows,10,1e-15)

    def test_ignores_augmented_terms(self):
        rows=[dict(q2=12.,g2=4.,d2=1e20,f2=1e20)]
        self.assertEqual(estimate_budget(rows,.01)['K_hat'],100)

    def test_actual_uniform_sampler_matches_estimator(self):
        from transport import create_transport, Sampler
        sampler=Sampler(create_transport(path_type='Linear',prediction='velocity'))
        calls=[]
        def model(x,t):
            calls.append(t.clone())
            return 2*x
        sample=sampler.sample_sde(sampling_method='Euler',diffusion_form='sigma',
                                 diffusion_norm=1.0,last_step=None,num_steps=6)
        sample(torch.ones(1,1,2,2),model)
        self.assertEqual(len(calls),5)

    def test_invalid_stabilization(self):
        for epsilon in [float('nan'),float('inf'),-1]:
            with self.assertRaises(ValueError):
                estimate_budget([dict(q2=3.,g2=1.)],.01,epsilon=epsilon)

    def test_budget(self):
        rows=[dict(q2=3.,d2=0.,f2=0.,g2=1.)]
        r=estimate_budget(rows,.11,epsilon=1e-15)
        self.assertEqual(r['K_hat'],10)
        self.assertEqual(r['NFE_hat_current_sampler'],10)
        self.assertEqual(r['NFE_hat_ideal'],10)
        self.assertGreater(r['previous_relative_error'],.11)
        self.assertIsNone(estimate_budget(rows,.01,max_k=5)['K_hat'])
        self.assertEqual(estimate_budget(rows,2)['K_hat'],1)

if __name__=='__main__':
    unittest.main()
