import copy
from dataclasses import fields
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from accelerate import Accelerator
from tqdm import tqdm

from lmflow.models.hf_text_regression_model import HFTextRegressionModel
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.dpov2_aligner import DPOv2Aligner
from lmflow.pipeline.rm_inferencer import RewardModelInferencer
from lmflow.pipeline.vllm_inferencer import MemorySafeVLLMInferencer
from lmflow.args import (
    ModelArguments, 
    DatasetArguments, 
    InferencerArguments,
    IterativeDPOAlignerArguments,
    DPOv2AlignerArguments,
)

logger = logging.getLogger(__name__)


class IterativeDPOAligner:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DatasetArguments,
        aligner_args:IterativeDPOAlignerArguments,
        ref_model_args: ModelArguments,
        reward_model_args: ModelArguments,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.aligner_args = aligner_args
        self.ref_model_args = ref_model_args
        self.reward_model_args = reward_model_args
        self.workspace_path = Path(self.aligner_args.output_dir)
        
        
    def align(
        self,
        dataset_list: List[Dataset]
    ):
        num_iterations = len(dataset_list)
        iteration_names = [f"iteration_{i+1}" for i in range(num_iterations)]
        
        for iter_idx, iter_name in tqdm(
            enumerate(iteration_names), 
            desc="Iterative DPO Align", 
            total=num_iterations,
            unit="iteration"
        ):
            if iter_idx == 0:
                target_model_args = self.model_args
            else:
                target_model_args = copy.deepcopy(self.model_args)
                target_model_args.model_name_or_path = str(self.workspace_path/f"iteration_{iter_idx-1}"/"model")
                
            self._align_single_iteration(
                iteration_name=iter_name,
                target_model_args=self.model_args,
                reward_model_args=self.reward_model_args,
                ref_model_args=self.ref_model_args,
                dataset=dataset_list[iter_idx],
            )
    
    
    def _align_single_iteration(
        self,
        iteration_name: str,
        target_model_args: ModelArguments,
        reward_model_args: ModelArguments,
        ref_model_args: ModelArguments,
        dataset: Dataset,
    ):
        if Accelerator().is_main_process:
            model = HFDecoderModel(
                model_args=target_model_args,
                tune_strategy='none'
            )
            self._do_target_model_inference(
                model=model,
                dataset=dataset,
                output_dir=str(self.workspace_path/iteration_name),
            )
            del model
        Accelerator().wait_for_everyone()
        
        reward_model = HFTextRegressionModel(
            model_args=reward_model_args,
            tune_strategy='none',
            use_accelerator=self.aligner_args.use_accelerator,
        )
        target_model_inference_result_data_args = copy.deepcopy(dataset.data_args)
        target_model_inference_result_data_args.dataset_path = str(self.workspace_path/iteration_name/"target_model_inference_result"/"result.json")
        target_model_inference_result_dataset = Dataset(target_model_inference_result_data_args)
        self._do_reward_model_inference(
            model=reward_model,
            dataset=target_model_inference_result_dataset,
            output_dir=str(self.workspace_path/iteration_name),
        )
        del reward_model
        
        target_model = HFDecoderModel(target_model_args)
        ref_model = HFDecoderModel(ref_model_args)
        dpo_train_data_args = copy.deepcopy(dataset.data_args)
        dpo_train_data_args.dataset_path = str(self.workspace_path/iteration_name/"reward_model_inference_result"/"result.json")
        dpo_train_dataset = Dataset(dpo_train_data_args)
        dpo_eval_dataset = copy.deepcopy(dpo_train_dataset.sample(
            n=100, 
            seed=self.aligner_args.random_seed
        ))
        self._do_single_dpo_align(
            model=target_model,
            ref_model=ref_model,
            train_dataset=dpo_train_dataset,
            eval_dataset=dpo_eval_dataset,
            output_dir=str(self.workspace_path/iteration_name/"model"),
        )
        del target_model
        del ref_model
    
    
    def _do_target_model_inference(
        self,
        model: HFDecoderModel,
        dataset: Dataset,
        output_dir: str,
    ):
        result_cache_path = str(Path(output_dir)/"cache"/"target_model_inference_result.json")
        inferencer = MemorySafeVLLMInferencer(
            model_args=model.model_args,
            data_args=dataset.data_args,
            inferencer_args=self._parse_target_model_inference_args(
                args=self.aligner_args,
                result_cache_path=result_cache_path,
            ),
        )
        res = inferencer.inference()
        
        dataset_out = {"type": "text_to_textlist", "instances": []}
        inferencer_inputs = model.prepare_inputs_for_inference(
            dataset,
            apply_chat_template=True,
            use_vllm=True,
        )
        for idx, instance in enumerate(inferencer_inputs):
            dataset_out["instances"].append({
                "input": instance,
                "output": res[idx],
            })
            
        json.dump(
            dataset_out, 
            open(str(Path(output_dir)/"target_model_inference_result"/"result.json"), "w", encoding='utf-8'),
            ensure_ascii=False,
            indent=4,
        )
        
        
    def _do_reward_model_inference(
        self,
        model: HFTextRegressionModel,
        dataset: Dataset,
        output_dir: str,
    ):
        inferencer = RewardModelInferencer(
            model_args=model.model_args,
            data_args=dataset.data_args,
            inferencer_args=self._parse_reward_model_inference_args(self.aligner_args),
        )
        res = inferencer.inference(
            model=model,
            dataset=dataset,
            transform_dataset_in_place=True,
            use_vllm=False,
        )
        
        res.save(str(Path(output_dir)/"reward_model_inference_result"/"result.json"))
    
    
    def _do_single_dpo_align(
        self,
        model: HFDecoderModel,
        ref_model: HFDecoderModel,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        output_dir: str,
    ):
        aligner = DPOv2Aligner(
            model_args=model.model_args,
            data_args=train_dataset.data_args,
            aligner_args=self._parse_dpo_aligner_args(
                args=self.aligner_args,
                output_dir=output_dir,
            ),
            ref_model_args=ref_model.model_args,
        )
        aligner.align(
            model=model,
            ref_model=ref_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
    
    def _parse_target_model_inference_args(
        self,
        args: IterativeDPOAlignerArguments,
        result_cache_path: str,
    ) -> InferencerArguments:
        inferencer_args = self.__filter_args(
            mixed_args=args,
            target_cls=InferencerArguments,
        )
        inferencer_args.save_results=True
        inferencer_args.results_path=result_cache_path
        inferencer_args.enable_distributed_vllm_inference=True
        
        return inferencer_args
    
    
    def _parse_reward_model_inference_args(
        self,
        args: IterativeDPOAlignerArguments,
    ) -> InferencerArguments:
        inferencer_args = self.__filter_args(
            mixed_args=args,
            target_cls=InferencerArguments,
        )
        
        return inferencer_args
    
    
    def _parse_dpo_aligner_args(
        self,
        args: IterativeDPOAlignerArguments,
        output_dir: str,
    ) -> DPOv2AlignerArguments:
        aligner_args = self.__filter_args(
            mixed_args=args,
            target_cls=DPOv2AlignerArguments,
        )
        aligner_args.output_dir = output_dir
        
        return aligner_args
    
    
    def __filter_args(
        self,
        mixed_args,
        target_cls,
    ):
        target_cls_fields = {f.name for f in fields(target_cls)}
        common_fields = {f: getattr(mixed_args, f) for f in target_cls_fields if hasattr(mixed_args, f)}
        return target_cls(**common_fields)