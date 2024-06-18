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
        self.eos_token_id = AutoTokenizer.from_pretrained(model_args.model_name_or_path).eos_token_id

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
        assert model_args.use_vllm_inference, "The model_args.use_vllm_inference must be True."
        super().__init__(model_args, inferencer_args)
        self.sampling_params = self.parse_to_sampling_params(inferencer_args)
        
    
    def parse_to_sampling_params(
        self,
        inference_args: InferencerWithOffloadingArguments,
    ) -> SamplingParams:
        return SamplingParams(
            use_beam_search=inference_args.use_beam_search,
            n=inference_args.num_output_sequences,
            temperature=inference_args.temperature,
            max_tokens=inference_args.max_new_tokens,
            seed=inference_args.seed,
            top_p=inference_args.top_p,
            top_k=inference_args.top_k,
            stop_token_ids=[self.eos_token_id] + inference_args.additional_stop_token_ids
        )


    def inference(
        self,
        model: HFDecoderModel, 
        dataset: Dataset, 
        apply_chat_template: bool = True,
        detokenize: bool = False,
        release: bool = False,
        inference_args: Optional[InferencerWithOffloadingArguments] = None,
        save_file_path: Optional[str] = None,
    ) -> Union[List[List[str]], List[List[List[int]]]]:
        """Perform inference using the provided model and dataset.

        Parameters
        ----------
        model : HFDecoderModel
            LMFlow HFDecoderModel object
        dataset : Dataset
            LMFlow Dataset object
        apply_chat_template : bool, optional
            Whether to apply chat template to the input, by default True.
        detokenize : bool, optional
            Whether to decode after generation, by default False.
        release : bool, optional
            Whether to release gpu resources, by default False. 
            NOTE: The reason why `release` and `detokenize` are not in `inference_args` is that
            Inferencer may be used by other pipeline, and the pipeline may want to control these 
            two behaviors dynamically. 
        inference_args : Optional[InferencerWithOffloadingArguments], optional
            by default None
        save_file_path : Optional[str], optional
            A json **file** path, by default None. If specified, the inference result will 
            be saved to the file.

        Returns
        -------
        Union[List[List[str]], List[List[List[int]]]]
            When `detokenize = True`, return a list of list of strings. Inner list
            contains inference_args.num_output_sequences samples for a single prompt 
            (i.e., `len(res[i]) = inference_args.num_output_sequences`). Outer list 
            contains the results for all prompts (i.e., `len(res) = len(dataset)`).
            
            When `detokenize = False`, return a list of list of list of ints 
            (token ids, no decoding after generation).
        """
        if save_file_path:
            # Since other pipeline may use the saved json file for further tasks,
            # use assert here rather than a warning.
            assert save_file_path.endswith(".json"), "The save_file_path must be a json file."
        if inference_args:
            logger.warning(
                "Overriding the default inference arguments with the provided arguments in .inference()"
            )
            sampling_params = self.parse_to_sampling_params(inference_args)
        else:
            sampling_params = self.sampling_params
            
        sampling_params.detokenize = detokenize
        
        model_input = model.prepare_inputs_for_inference(
            dataset=dataset, 
            apply_chat_template=apply_chat_template
        )
        
        outputs = model.inference(
            inputs=model_input,
            sampling_params=sampling_params,
            release=release,
        )
                
        if save_file_path:
            self.save_inference_results(outputs, save_file_path)
                    
        return outputs
    
    
    def save_inference_results(
        self,
        outputs: Union[List[List[str]], List[List[List[int]]]],
        save_file_path: str,
    ):
        with open(save_file_path, "w") as f:
            json.dump(outputs, f)
            
        logger.info(f"Inference results are saved to {save_file_path}.")