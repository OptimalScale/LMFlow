from lmflow.pipeline.inferencer import ToolInferencer
import unittest
from lmflow.args import InferencerArguments
from lmflow.args import ModelArguments
from lmflow.args import DatasetArguments
from lmflow.models import hf_decoder_model

CODE_1 = "print(\"hello world\")"
RES_1 = "hello world\n"
CODE_2 = "b=a+1\nprint(b)"
RES_2 = """Traceback (most recent call last):
  File "<string>", line 1, in <module>
NameError: name 'a' is not defined
"""

class ToolInferencerTest(unittest.TestCase):
    def set_up(self):
        model_args = ModelArguments(model_name_or_path="codellama/CodeLlama-7b-instruct-hf")
        model = hf_decoder_model.HFDecoderModel(model_args)
        inferencer_args = InferencerArguments()
        data_args = DatasetArguments()
        self.toolinf = ToolInferencer(model_args, data_args, inferencer_args)
        
    def test_code_exec_1(self,code=CODE_1, expected_output=RES_1):
        
        toolinf_res = self.toolinf.code_exec(code)
        self.assertEqual(toolinf_res, expected_output)
        
    def test_code_exec_2(self,code=CODE_2):
        toolinf_res = self.toolinf.code_exec(code)
        self.assertNotEqual(toolinf_res.returncode, 0)
        
unittest.main()
        
        
        