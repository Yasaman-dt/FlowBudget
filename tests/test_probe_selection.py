import unittest
from pathlib import Path
from flowbudget_config import load_config
from probe_sensitivity import select_models

class SelectionTests(unittest.TestCase):
    def test_subsets_and_solver_preservation(self):
        root=Path(__file__).resolve().parents[1]
        config=load_config(root/'configs/probe_sensitivity.json')
        self.assertEqual(len(config['pairs']),6)
        self.assertEqual(select_models(config,['SiT-XL/2'])['pairs'],config['pairs'][4:])
        self.assertEqual(select_models(config,['RF-UNet','3-RF','2-RF','1-RF'])['pairs'],config['pairs'][:4])
        self.assertEqual(len(config['pairs']),6)
        for selection in [[],['missing'],['1-RF','1-RF']]:
            with self.assertRaises(ValueError):select_models(config,selection)
