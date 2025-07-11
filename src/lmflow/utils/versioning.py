import importlib
import logging
import sys
from pathlib import Path
from typing import Union

import pkg_resources

logger = logging.getLogger(__name__)


def get_python_version():
    return sys.version_info


def _is_package_available(package_name: str, skippable: bool = False):
    assert isinstance(package_name, str), f"Invalid type of package_name: {type(package_name)}"
    try:
        importlib.import_module(package_name)
        return True
    except Exception as e:
        if isinstance(e, ModuleNotFoundError):
            return False
        else:
            if skippable:
                logger.warning(f"An error occurred when importing {package_name}:\n{e}\n{package_name} is disabled.")
                return False
            else:
                raise e


def _is_packages_available(packages: Union[list[str], list[tuple[str, bool]]]):
    if isinstance(packages[0], str):
        return all([_is_package_available(package) for package in packages])
    elif isinstance(packages[0], tuple):
        return all([_is_package_available(package, skippable) for package, skippable in packages])
    else:
        raise ValueError(f"Invalid type of packages: {type(packages[0])}")


def is_package_version_at_least(package_name, min_version):
    try:
        package_version = pkg_resources.get_distribution(package_name).version
        if pkg_resources.parse_version(package_version) < pkg_resources.parse_version(min_version):
            return False
    except pkg_resources.DistributionNotFound:
        return False
    return True


def is_gradio_available():
    return _is_package_available("gradio")


def is_ray_available():
    return _is_package_available("ray")


def is_vllm_available():
    return _is_package_available("vllm")


def is_flash_attn_available():
    return _is_package_available("flash_attn", skippable=True)


def is_flask_available():
    return _is_packages_available(["flask", "flask_cors"])


def is_trl_available():
    return _is_package_available("trl")


def is_multimodal_available():
    return _is_packages_available(["PIL"])


def is_deepspeed_available():
    return _is_package_available("deepspeed")


def get_lmflow_dir(return_src_dir: bool = False) -> Path:
    try:
        from importlib.util import find_spec

        spec = find_spec("lmflow")
        if spec is None:
            return None

        origin = Path(spec.origin)
        if origin is None:
            return None

        if origin.name == "__init__.py":
            if return_src_dir:
                return origin.parent
            else:
                return origin.parent.parent.parent
        else:
            return None

    except (ImportError, AttributeError):
        return None
