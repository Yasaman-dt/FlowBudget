import math
import unittest
import torch
from estimate_em_three_seeds import drift,coefficient

class Tests(unittest.TestCase):
    def test_matches_native_sde_drift(self):
        x=torch.randn(1,1,2,2,dtype=torch.float64)
        t=torch.tensor([.3],dtype=torch.float64)
        v=lambda x,t:2*x
        score=(t*v(x,t)-x)/(1-t)
        torch.testing.assert_close(drift(v,x,t),v(x,t)+.5*(1-t)*score)
    def test_linear_exact_coefficient(self):
        x=torch.ones(1,1,2,2,dtype=torch.float64)
        t=torch.tensor([.3],dtype=torch.float64)
        velocity=lambda x,t:2*x
        g=math.sqrt(.7); gp=-.5/g
        av=(2*(1+.5*.3)-.5)*g
        expected=math.sqrt(4*(av*av+gp*gp+av*gp))/(g*2+1e-8)
        result=coefficient(velocity,x,t,[torch.ones_like(x),-torch.ones_like(x)],[.0005,.001,.002])
        for values in result.values():self.assertAlmostEqual(values[0],expected,places=9)

if __name__=='__main__':unittest.main()
