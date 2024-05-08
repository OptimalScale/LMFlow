from .base import StringFormatter, TemplateComponent, ConversationTemplate


QWEN2_TEMPLATE = ConversationTemplate(
    template_name='qwen2',
    user_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>user\n{{content}}<|im_end|>\n')
        ]
    ),
    assistant_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>assistant\n{{content}}<|im_end|>\n')
        ]
    ),
    system_formatter=StringFormatter(
        template=[
            TemplateComponent(type='string', content='<|im_start|>system\n{{content}}<|im_end|>\n')
        ]
    ),
    separator=TemplateComponent(type='string', content='\n')
)