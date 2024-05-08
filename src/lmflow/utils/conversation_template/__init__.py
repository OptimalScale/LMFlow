from .base import EMPTY_TEMPLATE, EMPTY_NO_SPECIAL_TOKENS_TEMPLATE, ConversationTemplate
from .chatml import CHATML_TEMPLATE
from .deepseek import DEEPSEEK_TEMPLATE
from .llama import LLAMA2_TEMPLATE, LLAMA3_TEMPLATE
from .phi import PHI3_TEMPLATE
from .qwen import QWEN2_TEMPLATE


PRESET_TEMPLATES = {
    'chatml': CHATML_TEMPLATE,
    'deepseek': DEEPSEEK_TEMPLATE,
    'empty': EMPTY_TEMPLATE,
    'empty_no_special_tokens': EMPTY_NO_SPECIAL_TOKENS_TEMPLATE,
    'llama2': LLAMA2_TEMPLATE,
    'llama3': LLAMA3_TEMPLATE,
    'phi3': PHI3_TEMPLATE,
    'qwen2': QWEN2_TEMPLATE,
}