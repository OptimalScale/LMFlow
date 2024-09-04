import copy
from dataclasses import fields
import gc
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from lmflow.models.hf_text_regression_model import HFTextRegressionModel
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.dpov2_aligner import MemorySafeDPOv2Aligner
from lmflow.pipeline.rm_inferencer import RewardModelInferencer
from lmflow.pipeline.vllm_inferencer import MemorySafeVLLMInferencer
from lmflow.args import (
    ModelArguments, 
    DatasetArguments, 
    InferencerArguments,
    IterativeDPOAlignerArguments,
    DPOv2AlignerArguments,
)
from lmflow.utils.common import print_banner

logger = logging.getLogger(__name__)


class IterativeDPOAligner:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DatasetArguments,
        aligner_args:IterativeDPOAlignerArguments,
        ref_model_args: ModelArguments,
        reward_model_args: ModelArguments,
        **kwargs,
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
        
        for iter_idx in tqdm(
            range(self.aligner_args.initial_iter_idx, num_iterations), 
            desc="Iterative DPO Align", 
            unit="iteration"
        ):
            if iter_idx == 0:
                target_model_args = self.model_args
            else:
                target_model_args = copy.deepcopy(self.model_args)
                target_model_args.model_name_or_path = str(self.workspace_path/f"iteration_{iter_idx}"/"model")
                
            self._align_single_iteration(
                iteration_name=f"iteration_{iter_idx+1}",
                target_model_args=target_model_args,
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
        if self.aligner_args.do_response_generation:
            # generate responses
            print_banner(f'Iterative DPO {iteration_name}: Generate responses')
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
        
        if self.aligner_args.do_scoring:
        # reward model scoring
            print_banner(f'Iterative DPO {iteration_name}: Reward model scoring')
            reward_model = HFTextRegressionModel(
                model_args=reward_model_args,
                tune_strategy='none',
                use_accelerator=self.aligner_args.use_accelerator,
            )
            target_model_inference_result_data_args = copy.deepcopy(dataset.data_args)
            target_model_inference_result_data_args.dataset_path = str(self.workspace_path/iteration_name/"target_model_inference_result")
            target_model_inference_result_data_args.block_size = self.aligner_args.reward_model_inference_block_size
            target_model_inference_result_dataset = Dataset(target_model_inference_result_data_args)
            self._do_reward_model_inference(
                model=reward_model,
                dataset=target_model_inference_result_dataset,
                output_dir=str(self.workspace_path/iteration_name),
            )
            del reward_model
        
        if self.aligner_args.do_dpo_align:
            # DPO training
            print_banner(f'Iterative DPO {iteration_name}: DPO training')
            dpo_train_data_args = copy.deepcopy(dataset.data_args)
            dpo_train_data_args.dataset_path = str(self.workspace_path/iteration_name/"reward_model_inference_result")
            self._do_single_dpo_align(
                model_args=target_model_args,
                ref_model_args=ref_model_args,
                data_args=dpo_train_data_args,
                output_dir=str(self.workspace_path/iteration_name/"model"),
                iteration_name=iteration_name,
            )
    
    
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
        
        dataset_out = {"type": "text_to_textlist", "instances": res}
            
        target_model_inference_result_dir = Path(output_dir)/"target_model_inference_result"
        target_model_inference_result_dir.mkdir(parents=True, exist_ok=True)
        json.dump(
            dataset_out, 
            open(str(target_model_inference_result_dir/"result.json"), "w", encoding='utf-8'),
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
            enable_distributed_inference=self.aligner_args.enable_distributed_inference,
            distributed_inference_num_instances=self.aligner_args.distributed_inference_num_instances,
            inference_batch_size=self.aligner_args.reward_model_inference_batch_size,
        )
        
        reward_model_inference_result_dir = Path(output_dir)/"reward_model_inference_result"
        reward_model_inference_result_dir.mkdir(parents=True, exist_ok=True)
        res.save(str(reward_model_inference_result_dir/"result.json"))
    
    
    def _do_single_dpo_align(
        self,
        model_args: ModelArguments,
        ref_model_args: ModelArguments,
        data_args: DatasetArguments,
        output_dir: str,
        iteration_name: str,
    ):
        aligner = MemorySafeDPOv2Aligner(
            model_args=model_args,
            data_args=data_args,
            aligner_args=self._parse_dpo_aligner_args(
                args=self.aligner_args,
                output_dir=output_dir,
                iteration_name=iteration_name,
            ),
            ref_model_args=ref_model_args,
        )
        aligner.align()
        
    
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
        iteration_name: str,
    ) -> DPOv2AlignerArguments:
        aligner_args = self.__filter_args(
            mixed_args=args,
            target_cls=DPOv2AlignerArguments,
        )
        aligner_args.output_dir = output_dir
        aligner_args.run_name = f"{args.run_name}_{iteration_name}"
        
        return aligner_args
    
    
    def __filter_args(
        self,
        mixed_args,
        target_cls,
    ):
        target_cls_fields = {f.name for f in fields(target_cls) if f.init}
        common_fields = {f: getattr(mixed_args, f) for f in target_cls_fields if hasattr(mixed_args, f)}
        return target_cls(**common_fields)