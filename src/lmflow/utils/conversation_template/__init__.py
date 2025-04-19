#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.
import logging

from lmflow.utils.versioning import is_package_version_at_least

from .base import EMPTY_TEMPLATE, EMPTY_NO_SPECIAL_TOKENS_TEMPLATE, ConversationTemplate, ConversationTemplateForTool, StringFormatter, TemplateComponent
from .chatglm import CHATGLM3_TEMPLATE
from .chatml import CHATML_TEMPLATE
from .deepseek import (
    DEEPSEEK_V2_TEMPLATE,
    DEEPSEEK_V3_TEMPLATE,
    DEEPSEEK_R1_TEMPLATE,
    DEEPSEEK_R1_DISTILL_TEMPLATE
)
from .gemma import GEMMA_TEMPLATE
from .hymba import HYMBA_TEMPLATE
from .internlm import INTERNLM2_TEMPLATE
from .llama import LLAMA2_TEMPLATE, LLAMA3_TEMPLATE, LLAMA3_TEMPLATE_FOR_TOOL
from .phi import PHI3_TEMPLATE
from .qwen import (
    QWEN2_TEMPLATE,
    QWEN2_TEMPLATE_FOR_TOOL,
    QWEN2_5_TEMPLATE,
    QWEN2_5_1M_TEMPLATE,
    QWEN2_5_MATH_TEMPLATE,
    QWEN_QWQ_TEMPLATE
)
from .yi import YI1_5_TEMPLATE
from .zephyr import ZEPHYR_TEMPLATE


logger = logging.getLogger(__name__)


PRESET_TEMPLATES = {
    'chatglm3': CHATGLM3_TEMPLATE,
    'chatml': CHATML_TEMPLATE,
    'deepseek': DEEPSEEK_V2_TEMPLATE,
    'deepseek_v2': DEEPSEEK_V2_TEMPLATE,
    'disable': EMPTY_TEMPLATE,
    'empty': EMPTY_TEMPLATE,
    'empty_no_special_tokens': EMPTY_NO_SPECIAL_TOKENS_TEMPLATE,
    'gemma': GEMMA_TEMPLATE,
    'hymba': HYMBA_TEMPLATE,
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

JINJA_TEMPLATES = {
    'deepseek_r1': DEEPSEEK_R1_TEMPLATE,
    'deepseek_r1_distill': DEEPSEEK_R1_DISTILL_TEMPLATE,
    'deepseek_v3': DEEPSEEK_V3_TEMPLATE,
    'qwen2_5': QWEN2_5_TEMPLATE,
    'qwen2_5_1m': QWEN2_5_1M_TEMPLATE,
    'qwen2_5_math': QWEN2_5_MATH_TEMPLATE,
    'qwen_qwq': QWEN_QWQ_TEMPLATE,
}

if is_package_version_at_least("transformers", "4.43.0"):
    for template_name, template in JINJA_TEMPLATES.items():
        PRESET_TEMPLATES[template_name] = template
else:
    logger.warning(
        f"The following conversation templates require transformers>=4.43.0: {JINJA_TEMPLATES.keys()}. "
        f"Please upgrade `transformers` to use them."
    )