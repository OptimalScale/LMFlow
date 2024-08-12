#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
from .base import EMPTY_TEMPLATE, EMPTY_NO_SPECIAL_TOKENS_TEMPLATE, ConversationTemplate, ConversationTemplateForTool
from .chatglm import CHATGLM3_TEMPLATE
from .chatml import CHATML_TEMPLATE
from .deepseek import DEEPSEEK_TEMPLATE
from .gemma import GEMMA_TEMPLATE
from .internlm import INTERNLM2_TEMPLATE
from .llama import LLAMA2_TEMPLATE, LLAMA3_TEMPLATE, LLAMA3_TEMPLATE_FOR_TOOL
from .phi import PHI3_TEMPLATE
from .qwen import QWEN2_TEMPLATE, QWEN2_TEMPLATE_FOR_TOOL
from .yi import YI1_5_TEMPLATE
from .zephyr import ZEPHYR_TEMPLATE


PRESET_TEMPLATES = {
    'chatglm3': CHATGLM3_TEMPLATE,
    'chatml': CHATML_TEMPLATE,
    'deepseek': DEEPSEEK_TEMPLATE,
    'disable': EMPTY_TEMPLATE,
    'empty': EMPTY_TEMPLATE,
    'empty_no_special_tokens': EMPTY_NO_SPECIAL_TOKENS_TEMPLATE,
    'gemma': GEMMA_TEMPLATE,
    'internlm2': INTERNLM2_TEMPLATE,
    'llama2': LLAMA2_TEMPLATE,
    'llama3': LLAMA3_TEMPLATE,
    'llama3_for_tool': LLAMA3_TEMPLATE_FOR_TOOL,
    'phi3': PHI3_TEMPLATE,
    'qwen2': QWEN2_TEMPLATE,
    'qwen2_for_tool': QWEN2_TEMPLATE_FOR_TOOL,
    'yi': CHATML_TEMPLATE,
    'yi1_5': YI1_5_TEMPLATE,
    'zephyr': ZEPHYR_TEMPLATE
}