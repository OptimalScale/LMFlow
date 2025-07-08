import os


def is_accelerate_env():
    for key, _ in os.environ.items():
        if key.startswith("ACCELERATE_"):
            return True
    return False
