import json
from pathlib import Path
import logging
from typing import List, Union, Optional

from vllm import SamplingParams
from transformers import AutoTokenizer

from lmflow.datasets import Dataset
from lmflow.pipeline.base_pipeline import BasePipeline
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.args import InferencerWithOffloadingArguments, ModelArguments


logger = logging.getLogger(__name__)


class InferencerWithOffloading(BasePipeline):
    def __init__(
        self, 
        model_args: ModelArguments,
        inferencer_args: InferencerWithOffloadingArguments,
    ):
        self.model_args = model_args
        self.inferencer_args = inferencer_args
        self.seed = inferencer_args.seed
        self.eos_token_id = AutoTokenizer.from_pretrained(model_args.model_name_or_path).eos_token_id
        self.activated = False

    def inference(self):
        raise NotImplementedError(".inference is not implemented")
        
    def save(self):
        raise NotImplementedError(".save is not implemented")
    
    
class VLLMInferencer(InferencerWithOffloading):
    def __init__(
        self,
        model_args: ModelArguments,
        inferencer_args: InferencerWithOffloadingArguments,
    ):
        super().__init__(model_args, inferencer_args)
        self.sampling_params = SamplingParams(
            use_beam_search=self.inferencer_args.use_beam_search,
            n=self.inferencer_args.num_output_sequences,
            temperature=self.inferencer_args.temperature,
            max_tokens=self.inferencer_args.max_new_tokens,
            seed=self.seed,
            top_p=self.inferencer_args.top_p,
            top_k=self.inferencer_args.top_k,
            stop_token_ids=[self.eos_token_id] + self.inferencer_args.additional_stop_token_ids
        )


    def inference(
        self,
        model: HFDecoderModel, 
        dataset: Dataset, 
        detokenize: bool = False,
        release: bool = True,
        inference_args: Optional[InferencerWithOffloadingArguments] = None,
        save_file_path: Optional[str] = None,
    ) -> Union[List[List[str]], List[List[List[int]]]]:
        if save_file_path:
            assert save_file_path.endswith(".json"), "The save_file_path must be a json file."
        if inference_args:
            logger.warning("Overriding the default inference arguments with the provided arguments.")
            sampling_params = SamplingParams(
                use_beam_search=inference_args.use_beam_search,
                n=inference_args.num_output_sequences,
                temperature=inference_args.temperature,
                max_tokens=inference_args.max_new_tokens,
                seed=self.seed,
                top_p=inference_args.top_p,
                top_k=inference_args.top_k,
                stop_token_ids=[self.eos_token_id] + inference_args.additional_stop_token_ids
            )
        else:
            sampling_params = self.sampling_params
            
        sampling_params.detokenize = detokenize
        
        user_input = dataset.get_backend_dataset() # TODO
        
        outputs = model.vllm_inference(
            user_input=user_input,
            sampling_params=sampling_params,
            release=release,
        )
                
        if save_file_path:
            with open(save_file_path, "w") as f:
                json.dump(outputs, f)
                    
        return outputs