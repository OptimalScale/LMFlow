#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging
from dataclasses import dataclass, field, fields, Field, make_dataclass
from pathlib import Path
from typing import Optional, List, Union, Dict

from lmflow.utils.versioning import get_python_version

logger = logging.getLogger(__name__)


def make_shell_args_from_dataclass(
    dataclass_objects: List, 
    format: str="subprocess",
    skip_default: bool=True,
    ignored_args_list: Optional[List[str]]=None,
) -> Union[str, List[str]]:
    """Return a string or a list of strings that can be used as shell arguments.

    Parameters
    ----------
    dataclass_objects : List
        A list of dataclass objects.
    format : str, optional
        Return format, can be "shell" or "subprocess", by default "subprocess".
    skip_default : bool, optional
        Whether to skip attributes with default values, by default True. 

    Returns
    -------
    Union[str, List[str]]
    """
    assert isinstance(dataclass_objects, list), "dataclass_objects should be a list of dataclass objects."
    all_args = {}
    for dataclass_object in dataclass_objects:
        for k, v in dataclass_object.__dict__.items():
            if ignored_args_list and k in ignored_args_list:
                continue
            if k not in dataclass_object.__dataclass_fields__:
                # skip attributes that added dynamically
                continue
            if not v:
                # skip attributes with None values
                continue
            if skip_default:
                if dataclass_object.__dataclass_fields__[k].default == v:
                    continue
            
            if k not in all_args:
                if isinstance(v, Path):
                    all_args[k] = str(v)
                elif isinstance(v, list):
                    all_args[k] = ",".join(v)
                else:
                    all_args[k] = v
            elif k in all_args:
                if all_args[k] == v:
                    continue
                else:
                    logger.warning(f"Found different values for the same key: {k}, using value: {v} instead.")
                    all_args[k] = v
    
    if format == "shell":
        final_res = " ".join([f"--{k} {v}" for k, v in all_args.items()])
    elif format == "subprocess":
        final_res = []
        for k, v in all_args.items():
            final_res.extend([f"--{k}", str(v)])
    else:
        raise ValueError(f"Unknown format: {format}")
        
    return final_res


def create_copied_dataclass(
    original_dataclass, 
    field_prefix: str, 
    class_prefix: str, 
    new_default: Dict=None
):
    """Create a copied dataclass with new field names and default values.

    Parameters
    ----------
    original_dataclass : dataclass
    field_prefix : str
        The prefix to add to the **field** names of the copied dataclass.
    class_prefix : str
        The prefix to add to the **class** name of the copied dataclass.
    new_default : Dict, optional
        The new default values for the copied dataclass. When None, the 
        default values of the original dataclass are used.

    Returns
    -------
    dataclass
    """
    original_fields = fields(original_dataclass)
    new_default = new_default or {}
    new_fields = []
    for field in original_fields:
        if get_python_version().minor >= 10:
            new_field = (
                f"{field_prefix}{field.name}", 
                field.type, 
                Field(
                    default=new_default.get(f"{field_prefix}{field.name}", field.default), 
                    default_factory=field.default_factory,
                    init=field.init,
                    repr=field.repr,
                    hash=field.hash,
                    compare=field.compare,
                    metadata=field.metadata,
                    kw_only=False, # add in py3.10: https://docs.python.org/3/library/dataclasses.html
                )
            )
        else:
            new_field = (
                f"{field_prefix}{field.name}", 
                field.type, 
                Field(
                    default=new_default.get(f"{field_prefix}{field.name}", field.default), 
                    default_factory=field.default_factory,
                    init=field.init,
                    repr=field.repr,
                    hash=field.hash,
                    compare=field.compare,
                    metadata=field.metadata,
                )
            )
            
        new_fields.append(new_field)
    copied_dataclass = make_dataclass(f"{class_prefix}{original_dataclass.__name__}", new_fields)
    return copied_dataclass


def remove_dataclass_attr_prefix(data_instance, prefix: str) -> Dict:
    """Remove the prefix from the attribute names of a dataclass instance.

    Parameters
    ----------
    data_instance : dataclass
    prefix : str
        The prefix to remove from the attribute names of the dataclass instance.

    Returns
    -------
    Dict
    """
    new_attributes = {}
    for field in fields(data_instance):
        attr_name = field.name
        attr_value = getattr(data_instance, attr_name)
        new_attr_name = f"{attr_name[len(prefix):]}"
        new_attributes[new_attr_name] = attr_value
    
    return new_attributes


def add_dataclass_attr_prefix(data_instance, prefix: str) -> Dict:
    """Add the prefix to the attribute names of a dataclass instance.

    Parameters
    ----------
    data_instance : dataclass
    prefix : str
        The prefix to add to the attribute names of the dataclass instance.

    Returns
    -------
    Dict
    """
    new_attributes = {}
    for field in fields(data_instance):
        attr_name = field.name
        attr_value = getattr(data_instance, attr_name)
        new_attr_name = f"{prefix}{attr_name}"
        new_attributes[new_attr_name] = attr_value
    
    return new_attributes


def print_banner(message: str):
    length = len(message) + 8
    border = "#" * length

    logger.info(border)
    logger.info(f"#   {message}   #")
    logger.info(border)