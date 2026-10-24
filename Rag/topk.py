# Experiment setting to change number of retrieved docs
from experiment import get_setting

DEFAULT_TOP_K = 10

def top_k_values(config):
    value = get_setting(config, "TopK")
    if value is None:
        return [None]
    return [int(k) for k in (value if isinstance(value, list) else [value])]

def top_k_label(top_k):
    return "" if top_k is None else f"_top{top_k}"

def resolve_top_k(top_k):
    return DEFAULT_TOP_K if top_k is None else top_k
