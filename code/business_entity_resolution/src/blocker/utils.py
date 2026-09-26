"""Small shared helpers: logging, seeding, devices, parameter budgets."""

import json
import os
import random
import time

import numpy as np

_T0 = time.time()


def log(msg):
    print(f'[{time.time() - _T0:8.1f}s] {msg}', flush=True)


def n_workers(value):
    return int(value) if value and int(value) > 0 else (os.cpu_count() or 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_device(name='auto'):
    import torch
    if name and name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def count_parameters(module):
    """Unique parameters (a tensor shared by two sub-modules is counted once)."""
    seen, total = set(), 0
    for p in module.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def check_param_budget(module, max_params, name):
    n = count_parameters(module)
    if max_params and n >= max_params:
        raise ValueError(f'{name}: {n:,} parameters, the limit is < {int(max_params):,}. '
                         f'Pick a smaller model in the config.')
    log(f'{name}: {n / 1e6:.1f}M parameters (limit {int(max_params) / 1e6:.0f}M)')
    return n


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# Multi-GPU (e.g. Kaggle "GPU T4 x2"): set once by the CLIs from runtime.multi_gpu.
MULTI_GPU = {'enabled': True}


def n_gpus():
    try:
        import torch
        return torch.cuda.device_count() if MULTI_GPU['enabled'] and torch.cuda.is_available() else 0
    except ImportError:
        return 0


def data_parallel(module):
    """torch.nn.DataParallel over every visible GPU when there are >= 2, else the module itself."""
    if n_gpus() > 1:
        import torch
        return torch.nn.DataParallel(module)
    return module


def apply_runtime(cfg):
    """Process-wide runtime switches from the config."""
    MULTI_GPU['enabled'] = bool(cfg.get('runtime', {}).get('multi_gpu', True))
