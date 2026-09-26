"""
Configuration: configs/blocking.yaml holds every default. A user config is
deep-merged on top of it, then `--set dotted.key=value` overrides (values are
parsed as YAML, so `--set pipelines.bert.top_k_source2=50` gives an int).
"""

import copy
import os

import yaml

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, 'configs', 'blocking.yaml')


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg, key, value):
    node = cfg
    parts = key.split('.')
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def get_dotted(cfg, key, default=None):
    node = cfg
    for p in key.split('.'):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def load_config(path=None, overrides=()):
    with open(DEFAULT_CONFIG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if path and os.path.abspath(path) != DEFAULT_CONFIG:
        with open(path, encoding='utf-8') as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for item in overrides or ():
        key, _, raw = item.partition('=')
        if not _:
            raise ValueError(f'--set expects key=value, got {item!r}')
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def save_config(cfg, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def add_config_args(ap):
    """Arguments shared by every CLI."""
    ap.add_argument('--config', default=DEFAULT_CONFIG, help='YAML config (merged over configs/blocking.yaml)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='override a config value, e.g. --set pipelines.bert.top_k_source2=30 (repeatable)')
    return ap
