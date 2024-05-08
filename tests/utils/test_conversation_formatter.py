import unittest
from lmflow.utils.conversation_template.base import StringFormatter, TemplateComponent


class StringFormatterTest(unittest.TestCase):

    def test_format_string_component(self):
        formatter = StringFormatter(
            template=[
                TemplateComponent(type='token', content='bos_token'),
                TemplateComponent(type='string', content='[INST] {{content}} [/INST]'),
                TemplateComponent(type='token', content='eos_token')
            ]
        )
        formatted_components = formatter.format(content='Who are you?')
        expected_components = [
            TemplateComponent(type='token', content='bos_token'),
            TemplateComponent(type='string', content='[INST] Who are you? [/INST]'),
            TemplateComponent(type='token', content='eos_token')
        ]
        self.assertEqual(formatted_components, expected_components)


if __name__ == '__main__':
    unittest.main()