import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Set, Sequence, Literal, Union, List, Optional
import logging


logger = logging.getLogger(__name__)
    

@dataclass
class TemplateComponent:
    type: Literal['token', 'token_id', 'string', 'tools']
    content: Union[str, int, List[str], List[int]]
    mask: Optional[bool] = True # for token specific masking, work in progress
    
    def __post_init__(self):
        assert self.content, "Content of the component cannot be empty."
        
        if self.type == 'tools':
            assert isinstance(self.content, list), (
                f"Content of tools component must be a list, got {type(self.content)}")
        elif self.type in ['token', 'string']:
            assert isinstance(self.content, str), (
                f"Content of string/token component must be a string, got {type(self.content)}")
        elif self.type == 'token_id':
            assert isinstance(self.content, int) or all(isinstance(token_id, int) for token_id in self.content), (
                f"Content of token_id component must be an integer or a list of integers.")
        else:
            raise ValueError(f"The type of the component must be either "
                             f"'token', 'string' or 'tools', got {self.type}")
            
    def __repr__(self) -> str:
        return f"TemplateComponent(type={self.type}, content={self.content})"
    
    def __str__(self) -> str:
        return f"{self.content}"


@dataclass
class Formatter(ABC):
    template: List[TemplateComponent] = field(default_factory=list)
    
    @abstractmethod
    def format(self, **kwargs) -> List[TemplateComponent]: ...
    
    def has_placeholder(self):
        flag = False
        for component in self.template:
            if component.type == 'string':
                if re.search(r"{{(.*?)}}", component.content):
                    flag = True
                    break
        return flag


@dataclass
class EmptyFormatter(Formatter):
    def __post_init__(self):
        if self.has_placeholder():
            raise ValueError("Empty formatter should not have placeholders.")
    
    def format(self, **kwargs) -> list:
        """Empty formatter for when no formatting is needed.
        This is useful when user has already applied formatting to the dataset.

        Returns
        -------
        list
            Original template.
        """
        return self.template
    

@dataclass
class StringFormatter(Formatter):
    def __post_init__(self):
        if not self.has_placeholder():
            raise ValueError("String formatter should have placeholders.")
    
    def format(self, **kwargs) -> list:
        """Format the string components with the provided keyword arguments. 
        Mostly used for formatting system prompt, user and assistant messages.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments containing values to replace in the template components.

        Returns
        -------
        list
            Formatted template.
        """
        formatted_template = []
        for component in self.template:
            if component.type == 'string':
                for key, value in kwargs.items():
                    templated = component.content.replace("{{" + key + "}}", value)
                    formatted_template.append(TemplateComponent(type='string', content=templated))
            else:
                formatted_template.append(component)
                
        print(formatted_template)
        return formatted_template

    
@dataclass
class ListFormatter(Formatter):
    def format(self, **kwargs) -> list:
        pass # Work in progress