import inspect
import textwrap
import time
from functools import wraps
from typing import Optional


def trace_calls(func):
    func.has_called = False
    
    def wrapper(*args, **kwargs):
        if not func.has_called:
            func.has_called = True
            stack = inspect.stack()
            print(f"Function '{func.__name__}' was called. Call stack:")
            for frame_info in stack[1:]:
                print(f" - Function '{frame_info.function}' in {frame_info.filename}, line {frame_info.lineno}")
        return func(*args, **kwargs)
    
    return wrapper


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__}: {end_time - start_time:.4f}s")
        return result
    return wrapper


def print_tabulate_with_header(tabulate_df, header: Optional[str] = None):
    if header:
        df_len = len(tabulate_df.split('\n')[0])
        print('\n')
        print('+' + '-'*(df_len-2) + '+')
        wrap_header = textwrap.wrap(header, df_len-4)
        for header in wrap_header:
            print("|" + header.center(df_len-2, ' ') + "|")
            
    print(tabulate_df)