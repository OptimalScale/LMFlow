from collections import OrderedDict
from typing import Dict, List, Tuple, Union, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import numpy as np


def compare_model(
    model_ref: "nn.Module", 
    model_trained: "nn.Module", 
    module_trained: Optional[List[str]] = None
) -> None:
    state_dict_ref = model_ref.state_dict()
    state_dict_trained = model_trained.state_dict()
    assert set(state_dict_ref.keys()) == set(state_dict_trained.keys())
    
    for name in state_dict_ref.keys():
        if module_trained is not None:
            if any([module in name for module in module_trained]):
                assert torch.allclose(state_dict_ref[name], state_dict_trained[name], rtol=1e-4, atol=1e-5) is False
        else:
            assert torch.allclose(state_dict_ref[name], state_dict_trained[name], rtol=1e-4, atol=1e-5) is True