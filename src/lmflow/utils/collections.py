#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
from typing import Dict, Union
from dataclasses import fields, Field, make_dataclass


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