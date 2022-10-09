import os

def validate_env_var(var_name: str, default: str, allowed_values: list) -> str:
    value = os.getenv(var_name, default)
    if not value in allowed_values:
        raise ValueError(f"Failed to load invalid ENV value for {var_name}. Must be one of "+ ' ,'.join(allowed_values))
    return value

# Model size to use for global fallback (facebook Reaserch's NNLB model)
OFFLINE_MODEL = os.getenv('OFFLINE_MODEL', 'small', ['disabled', 'small', 'big'])

# Significantly improves JA -> EN translation when set (JParaCrawl / Sugoi)
OFFLINE_ENABLED_JA_EN = os.getenv('OFFLINE_MODEL_JA_EN', 'small', ['disabled', 'small', 'big'])

