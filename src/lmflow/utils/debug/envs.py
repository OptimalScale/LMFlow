import os

from accelerate.utils import DistributedType


def get_distributed_type():
    distributed_type = DistributedType.DEEPSPEED if "ACCELERATE_USE_DEEPSPEED" in os.environ else DistributedType.NO
    return distributed_type