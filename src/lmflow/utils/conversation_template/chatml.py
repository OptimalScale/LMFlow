from .base import StringFormatter, TemplateComponent, ConversationTemplate


CHATML_TEMPLATE = ConversationTemplate(
    template_name='chatml',
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
    )
)