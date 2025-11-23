"""
Utilities for handling deprecated APIs and maintaining backwards compatibility.
"""

import functools
import inspect
import warnings
from typing import Any, Callable, Dict

__all__ = ['deprecated_args']


def deprecated_args(**deprecated_params: Dict[str, Any]):
    """
    Decorator to handle deprecated function arguments.
    
    Args:
        **deprecated_params: Mapping of deprecated argument names to their configuration.
            Each value should be a dict with:
            - 'replacement': Name of the new argument (optional)
            - 'mapper': Function to map old value to new value (optional)
            - 'message': Custom deprecation message (optional)
    
    Example:
        @deprecated_args(
            use_vllm={
                'replacement': 'inference_engine',
                'mapper': lambda x: 'vllm' if x else 'huggingface',
                'message': "use_vllm is deprecated. Use inference_engine='vllm' instead."
            }
        )
        def my_function(inference_engine='huggingface', **kwargs):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature to handle both args and kwargs
            sig = inspect.signature(func)
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Check for deprecated arguments in kwargs
            for old_arg, config in deprecated_params.items():
                if old_arg in kwargs:
                    old_value = kwargs.pop(old_arg)
                    
                    # Build deprecation message
                    if 'message' in config:
                        message = config['message']
                    else:
                        replacement = config.get('replacement', 'a different argument')
                        message = (
                            f"'{old_arg}' is deprecated and will be removed in a future version. "
                            f"Please use '{replacement}' instead."
                        )
                    
                    warnings.warn(message, DeprecationWarning, stacklevel=2)
                    
                    # Map old value to new argument if specified
                    if 'replacement' in config:
                        new_arg = config['replacement']
                        
                        # Apply mapper function if provided
                        if 'mapper' in config:
                            new_value = config['mapper'](old_value)
                        else:
                            new_value = old_value
                        
                        # Only set the new argument if it wasn't already provided
                        if new_arg not in kwargs:
                            kwargs[new_arg] = new_value
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator