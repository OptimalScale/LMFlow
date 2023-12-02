import os
import argparse
from lmflow.args import InferencerArguments
from lmflow.args import ModelArguments
from lmflow.args import DatasetArguments
from lmflow.models import hf_decoder_model
from lmflow.pipeline.inferencer import ToolInferencer
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', 
                        help='gpu id, currently speculative inference only support single gpu')
    parser.add_argument('--model', type=str, default='codellama/CodeLlama-7b-instruct-hf',
                        help='target code generation model name or path you  \
                            currently only supports huggingface decoder only models')
    params = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu
    
    model_args = ModelArguments(model_name_or_path=params.model)
    model = hf_decoder_model.HFDecoderModel(model_args)
    inferencer_args = InferencerArguments()
    data_args = DatasetArguments()
    
    toolinf = ToolInferencer(model_args, data_args, inferencer_args)
    
    while True:
        try:
            text = input("Tool Inference: ")
            toolinf_res = toolinf.inference(model, text)
            toolinf_res = toolinf_res.replace("<s>","")
            toolinf_res = toolinf_res.replace("</s>","")
            print('\n\nResult:')
            print(toolinf_res)
            print('\n\n')
            run_code = input("Run code? (y/n): ")
            if run_code == 'y':
                toolinf.code_exec(toolinf_res)
            if run_code == 'n':
                continue
            

        except EOFError:
            break

if __name__ == '__main__':
    main()