from .version import __version__ as internal_version

__version__ = internal_version

from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from lmflow import args, datasets, models, pipeline, utils

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")