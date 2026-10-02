"""Check EM launcher pool sizing, resuming, and reported NFE without model runs."""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from probe_sensitivity import main_em
from scripts.run_sit_em_2000_three_seeds import main as fixed_main


class EMLauncherTests(unittest.TestCase):
    def run_launcher(self, launcher, counts, extra):
        def estimate(rows, tau, epsilon):
            k = len(rows)
            return dict(K_hat=k, R_theta_em=1, relative_error=tau,
                        NFE_hat_ideal=k, NFE_hat_current_sampler=k)

        def generate(command, **kwargs):
            value = lambda flag: command[command.index(flag) + 1]
            Path(value('--output')).write_text(json.dumps(dict(
                estimator='SiT-Linear-EM-leading-order-v2',
                probes=[{}] * int(value('--num-samples')),
                metadata=dict(seed=int(value('--seed')), vae='ema', batch_size=1,
                              trace_probes=8, diffusion_norm=1., t_min=.01))))

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            argv = ['--output-dir', tmp, '--gpu', '0'] + extra
            with patch.dict('sys.modules', {'estimate_em_budget': types.SimpleNamespace(estimate_budget=estimate)}), patch('probe_sensitivity.subprocess.run', side_effect=generate) as run:
                launcher(argv)
                self.assertEqual(run.call_count, 3)
                manifest = json.loads((output/'experiment.json').read_text())
                self.assertEqual(manifest['counts'], counts)
                self.assertEqual(manifest['sampling'], f'Nested prefixes of one {max(counts)}-probe pool per seed')
                for row in json.loads((output/'stability.json').read_text()):
                    self.assertEqual(row['mean_NFE_current_sampler'], row['mean_K_hat'])
                launcher(argv)
                self.assertEqual(run.call_count, 3)  # Reuse matching saved pools.
                with self.assertRaises(ValueError):
                    main_em(['--output-dir', tmp, '--gpu', '0', '--counts', '7'])

    def test_custom_prefixes_and_resume(self):
        self.run_launcher(main_em, [2, 5], ['--counts', '5', '2'])

    def test_fixed_launcher_defaults(self):
        self.run_launcher(fixed_main, [2000], [])

    def test_invalid_counts_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'new'
            for counts in [['0'], ['-1'], ['2', '2']]:
                with self.assertRaises(SystemExit):
                    main_em(['--output-dir', str(output), '--counts'] + counts)
                self.assertFalse(output.exists())
