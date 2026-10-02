"""Portable resource paths and JSON configuration expansion for experiment runners."""
import json
import os
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent


def env_path(name, default):
    """Resolve an environment override; relative paths are repository-relative."""
    value = Path(os.path.expandvars(os.environ.get(name, str(default)))).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def resource_path(default):
    """Apply shared overrides to known input resources, preserving other paths."""
    path = Path(default)
    value = path.as_posix()
    if value.endswith('datasets/imagenet_2012/images/train'):
        return env_path('FLOWBUDGET_IMAGENET', REPO_ROOT / 'data/imagenet/train')
    if value.endswith('guided-diffusion/evaluations/evaluator.py') or value.endswith('rectified_flow_modern/evaluations/evaluator.py'):
        return env_path('FLOWBUDGET_EVALUATOR', path)
    if '/checkpoints/' in value:
        prefix, suffix = value.split('/checkpoints/', 1)
        model = Path(prefix).name
        names = {'rectified_flow_cifar10': 'FLOWBUDGET_RF_CHECKPOINTS',
                 'rectified_flow_modern': 'FLOWBUDGET_UNET_CHECKPOINTS',
                 'FlowDCN': 'FLOWBUDGET_FLOWDCN_CHECKPOINTS',
                 'dit_flow_matching': 'FLOWBUDGET_DIT_CHECKPOINTS',
                 'edm_cifar10': 'FLOWBUDGET_EDM_CHECKPOINTS'}
        if model in names:
            return env_path(names[model], Path(prefix) / 'checkpoints') / suffix
    if value.endswith('rectified_flow_cifar10/data'):
        return env_path('FLOWBUDGET_CIFAR10', path)
    if value.endswith('references/cifar10_test_10k.npz'):
        return env_path('FLOWBUDGET_CIFAR_REFERENCE', path)
    if value.endswith('references/imagenet_train_10k_center_crop_256.npz'):
        return env_path('FLOWBUDGET_IMAGENET_REFERENCE', path)
    return path


def gpu_default():
    """Preserve an existing CUDA selection, including an empty CPU selection."""
    return os.environ.get('CUDA_VISIBLE_DEVICES', os.environ.get('FLOWBUDGET_GPU', '0'))


def select_gpu():
    """Set visibility before CUDA initialization; return the selected device list."""
    gpu = gpu_default()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    return gpu


def expand_config(value):
    """Expand resource placeholders without changing the source JSON on disk."""
    defaults = {
        'FLOWBUDGET_ROOT': REPO_ROOT,
        'FLOWBUDGET_IMAGENET': resource_path('datasets/imagenet_2012/images/train'),
        'FLOWBUDGET_CIFAR10': resource_path(REPO_ROOT / 'rectified_flow_cifar10/data'),
    }
    for model, name in [('rectified_flow_cifar10', 'RF'), ('rectified_flow_modern', 'UNET'),
                        ('FlowDCN', 'FLOWDCN'), ('dit_flow_matching', 'DIT'), ('edm_cifar10', 'EDM')]:
        defaults[f'FLOWBUDGET_{name}_CHECKPOINTS'] = env_path(
            f'FLOWBUDGET_{name}_CHECKPOINTS', REPO_ROOT / model / 'checkpoints')
    if isinstance(value, dict):
        return {key: expand_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_config(item) for item in value]
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if name in defaults:
                return str(defaults[name])
            if name in os.environ:
                return os.environ[name]
            raise ValueError(f'Undefined configuration variable: {name}')
        return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', replace, value)
    return value


def load_config(path):
    return expand_config(json.loads(Path(path).read_text()))


def optional_path(name):
    return env_path(name, os.environ[name]) if os.environ.get(name) else None


def device_default(cuda_available):
    return os.environ.get('FLOWBUDGET_DEVICE', 'cuda' if cuda_available else 'cpu')
