"""Portable settings must work across directories and preserve explicit choices."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from flowbudget_config import REPO_ROOT, expand_config, gpu_default, resource_path


class RuntimeConfigTests(unittest.TestCase):
    def test_nested_config_and_spaces(self):
        with patch.dict(os.environ, {'FLOWBUDGET_IMAGENET': '/tmp/data with spaces',
                                     'FLOWBUDGET_RF_CHECKPOINTS': 'weights/rf'}, clear=True):
            value = expand_config({'args': ['${FLOWBUDGET_IMAGENET}',
                         '${FLOWBUDGET_RF_CHECKPOINTS}/model.pth'], 'seed': 2})
            self.assertEqual(value['args'], ['/tmp/data with spaces',
                               str(REPO_ROOT / 'weights/rf/model.pth')])
            self.assertEqual(value['seed'], 2)
            self.assertEqual(resource_path(REPO_ROOT / 'rectified_flow_cifar10/checkpoints/model.pth'),
                             REPO_ROOT / 'weights/rf/model.pth')

    def test_gpu_inheritance_including_cpu(self):
        for visible in ['4', '1,3', '']:
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': visible, 'FLOWBUDGET_GPU': '7'}, clear=True):
                self.assertEqual(gpu_default(), visible)
        with patch.dict(os.environ, {'FLOWBUDGET_GPU': '7'}, clear=True):
            self.assertEqual(gpu_default(), '7')

    def test_unknown_token_fails_early(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Undefined'):
                expand_config('${MISSING_DATASET}')

    def test_config_loading_outside_repo(self):
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), FLOWBUDGET_IMAGENET=temp)
            code = ('import json; from flowbudget_config import load_config; '
                    f'print(json.dumps(load_config({str(REPO_ROOT / "configs/probe_sensitivity.json")!r})))')
            result = subprocess.run([sys.executable, '-c', code], cwd=temp, env=env,
                                    check=True, text=True, capture_output=True)
            config = json.loads(result.stdout)
            self.assertNotIn('${', result.stdout)
            self.assertTrue(any(temp in pair['args'] for pair in config['pairs']))

    def test_shell_launcher_preserves_device_and_interpreter(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = Path(temp) / 'python recorder'
            recorder.write_text('#!/usr/bin/env python3\nimport json,os,sys\nprint(json.dumps(dict(args=sys.argv[1:],gpu=os.environ["CUDA_VISIBLE_DEVICES"],cwd=os.getcwd())))\n')
            recorder.chmod(0o755)
            env = dict(os.environ, FLOWBUDGET_PYTHON=str(recorder), CUDA_VISIBLE_DEVICES='5')
            result = subprocess.run(['bash', str(REPO_ROOT / 'scripts/run_probe_sensitivity_20000.sh'), 'rf'],
                                    cwd=temp, env=env, text=True, capture_output=True, check=True)
            call = json.loads(result.stdout)
            self.assertEqual(call['gpu'], '5')
            self.assertEqual(call['args'][call['args'].index('--gpu')+1], '5')
            self.assertEqual(call['cwd'], str(REPO_ROOT))
