import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', 
                        help='gpu id, currently speculative inference only support single gpu')
    parser.add_argument('--model', type=str, default='gpt2-xl', 
                        help='target model name or path (i.e., the large model you want to accelerate), \
                            currently only supports huggingface decoder only models')
    parser.add_argument('--draft_model', type=str, default='gpt2',
                        help='draft model name or path, currently only supports huggingface decoder only models')
    parser.add_argument('--gamma', type=int, default=5,
                        help='number of tokens that the draft model will generate at each step')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='maximum number of tokens that the speculative inference will generate')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='temperature for sampling')
    
    params = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu
    from lmflow.args import InferencerArguments
    from lmflow.args import ModelArguments
    from lmflow.args import DatasetArguments
    from lmflow.models import hf_decoder_model
    from lmflow.pipeline.inferencer import SpeculativeInferencer
    
    
    model_args = ModelArguments(model_name_or_path=params.model)
    model = hf_decoder_model.HFDecoderModel(model_args)
    draft_model_args = ModelArguments(model_name_or_path=params.draft_model)
    draft_model = hf_decoder_model.HFDecoderModel(draft_model_args)
    inferencer_args = InferencerArguments()
    data_args = DatasetArguments()
    
    specinf = SpeculativeInferencer(model_args, draft_model_args, data_args, inferencer_args)
    
    while True:
        try:
            text = input("Speculative Inference: ")
            specinf_res = specinf.inference(model, 
                                            draft_model, 
                                            text, 
                                            gamma=params.gamma, 
                                            max_new_tokens=params.max_new_tokens, 
                                            temperature=params.temperature)
            print(specinf_res)
            print('\n\n')

        except EOFError:
            break
