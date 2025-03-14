import json
import os
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional

import pytest
import torch
from transformers.testing_utils import (
    TestCasePlus, 
    execute_subprocess_async,
    get_torch_dist_unique_port,
    _RunOutput,
)

from lmflow.utils.versioning import get_lmflow_dir
from lmflow.utils.versioning import is_deepspeed_available


TEST_START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
KEEP_TEST_FILES = os.environ.get("LMFLOW_KEEP_TEST_FILES", "1") == "1"
LOG_TEST_STD = os.environ.get("LMFLOW_LOG_TEST_STD", "1") == "1"
TEST_MODEL_NAME_OR_PATH = os.getenv("LMFLOW_TEST_MODEL_NAME_OR_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
TEST_STEPS = int(os.getenv("LMFLOW_TEST_STEPS", 100))
# python subprocess takes list of strings as command, don't forget to split
ACCELERATE_ARGS = f'''
    --machine_rank 0
    --main_training_function main
    --num_machines 1
    --num_processes {torch.cuda.device_count()}
    --rdzv_backend static
    --same_network
    --mixed_precision no
    --dynamo_backend no
    --main_process_port {get_torch_dist_unique_port()}
'''.split()
ACCELERATE_DSZ3_ARGS = f'''
    --use_deepspeed
    --deepspeed_multinode_launcher standard
    --offload_optimizer_device none
    --offload_param_device none
    --zero3_init_flag true
    --zero3_save_16bit_model true
    --zero_stage 3
'''.split()
ACCELERATE_FSDP_ARGS = f'''
    --use_fsdp
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP
    --fsdp_min_num_params 1000000
    --fsdp_backward_prefetch BACKWARD_PRE
    --fsdp_forward_prefetch false
    --fsdp_cpu_ram_efficient_loading true
    --fsdp_offload_params false
    --fsdp_sharding_strategy FULL_SHARD
    --fsdp_state_dict_type FULL_STATE_DICT
    --fsdp_sync_module_states true
    --fsdp_use_orig_params true
'''.split()
DEEPSPEED_BASE_ARGS = f'''
    --master_port {get_torch_dist_unique_port()}
    --num_gpus {torch.cuda.device_count()}
'''.split()
DEEPSPEED_ZERO3_CONFIG_ARGS = f'''
    --deepspeed configs/deepspeed/zero3.json
'''.split()
LMFLOW_BASE_ARGS = f'''
    --model_name_or_path {TEST_MODEL_NAME_OR_PATH}
    --trust_remote_code 0 
    --dataset_path {get_lmflow_dir()/"data/alpaca/train_conversation"} 
    --overwrite_output_dir 
    --conversation_template qwen2 
    --disable_group_texts 1 
    --num_train_epochs 1 
    --per_device_train_batch_size 1 
    --gradient_accumulation_steps 1 
    --block_size 512 
    --learning_rate 2e-5 
    --lr_scheduler_type cosine  
    --validation_split_percentage 0 
    --logging_steps 1 
    --max_steps {TEST_STEPS} 
    --do_train 
    --ddp_timeout 72000 
    --save_steps 5000 
    --use_flash_attention 0 
    --gradient_checkpointing 0 
    --dataloader_num_workers 8 
    --report_to wandb 
    --seed 42 
    --save_strategy no
'''.split()
LMFLOW_FP32_ARGS = f'''
    --torch_dtype float32
'''.split()
LMFLOW_BF16_ARGS = f'''
    --bf16
    --torch_dtype bfloat16
'''.split()
LMFLOW_LORA_ARGS = f'''
    --use_lora 1
    --lora_r 8
    --lora_alpha 32
    --lora_dropout 0.1
'''.split()
LMFLOW_QLORA_ARGS = f'''
    --use_qlora 1
    --quant_bit 4
    --lora_r 8
    --lora_alpha 32
    --lora_dropout 0.1
'''.split()
LMFLOW_LISA_ARGS = f'''
    --lisa_activated_layers 1
    --lisa_interval_steps 20
'''.split()
LMFLOW_CUSTOM_OPTIM_ARGS = f'''
    --use_customized_optim 1
    --customized_optim adabelief
    --optim_beta1 0.9
    --optim_beta2 0.99
    --optim_weight_decay 0
'''.split()


class AccelerateBackend(Enum):
    FSDP = "fsdp"
    DEEPSPEED_ZERO3 = "deepspeed_zero3"

class DeepSpeedZeroStage(Enum):
    ZERO3 = "zero3"

class PeftMethod(Enum):
    NO = "full"
    LORA = "lora"
    QLORA = "qlora"
    LISA = "lisa"
    
class TestDtype(Enum):
    FP32 = "fp32"
    BF16 = "bf16"

ACCELERATE_BACKEND_ARGS_MAPPING = {
    AccelerateBackend.FSDP: ACCELERATE_FSDP_ARGS,
    AccelerateBackend.DEEPSPEED_ZERO3: ACCELERATE_DSZ3_ARGS,
}
DEEPSPEED_ZERO_CONFIG_MAPPING = {
    DeepSpeedZeroStage.ZERO3: DEEPSPEED_ZERO3_CONFIG_ARGS,
}
LMFLOW_PEFT_ARGS_MAPPING = {
    PeftMethod.LORA: LMFLOW_LORA_ARGS,
    PeftMethod.QLORA: LMFLOW_QLORA_ARGS,
    PeftMethod.LISA: LMFLOW_LISA_ARGS,
}


class TestFinetunerBase(TestCasePlus):
    """Full finetune, and other base functionality tests"""
    def setUp(self):
        os.environ["WANDB_PROJECT"] = "lmflow-pytest"
        self.lmflow_dir = get_lmflow_dir()
        self.lmflow_examples_dir = self.lmflow_dir / "examples"
        return super().setUp()
    
    def _make_cmd(self, run_name: str, args: List[List[str]]) -> List[str]:
        cmd = []
        for arg in args:
            assert isinstance(arg, list)
            cmd.extend(arg)
            
        self.output_dir = self.get_auto_remove_tmp_dir(
            tmp_dir=f"./tests_out/{run_name}",
            before=True,
            after=not KEEP_TEST_FILES,
        )
        logging_args = f'''
        --run_name {TEST_START_TIME}_{run_name}
        --output_dir {self.output_dir}
        '''.split()
        
        return cmd + logging_args
    
    def _log_std(self, res: _RunOutput):
        if LOG_TEST_STD:
            with open(f"{self.output_dir}/stdout.log", "w") as f:
                for line in res.stdout:
                    f.write(line+"\n")
            with open(f"{self.output_dir}/stderr.log", "w") as f:
                for line in res.stderr:
                    f.write(line+"\n")
                
    def _load_trainer_state(self, output_dir: str) -> Dict:
        with open(f"{output_dir}/trainer_state.json", "r") as f:
            trainer_state = json.load(f)
        return trainer_state
                
    def _run_with_accelerate(
        self, 
        backend: AccelerateBackend, 
        *,
        dtype: Optional[TestDtype] = TestDtype.FP32,
        extra_args: Optional[List[str]] = None,
        run_name_note: Optional[str] = None
    ) -> Dict:
        assert isinstance(backend, AccelerateBackend)
        run_name = f"test_finetuner_accelerate_{backend.value}"
        if run_name_note:
            run_name += f"_{str(run_name_note)}"
        
        all_args = [
            ["accelerate", "launch"],
            ACCELERATE_ARGS,
            ACCELERATE_BACKEND_ARGS_MAPPING[backend],
            [f"{self.lmflow_examples_dir}/finetune.py"],
            LMFLOW_BASE_ARGS,
        ]
        if dtype == TestDtype.FP32:
            all_args.append(LMFLOW_FP32_ARGS)
        elif dtype == TestDtype.BF16:
            all_args.append(LMFLOW_BF16_ARGS)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
            
        if extra_args and len(extra_args) > 0:
            all_args.append(extra_args)
            
        cmd = self._make_cmd(run_name, all_args)
        res = execute_subprocess_async(cmd)
        self._log_std(res)
        return self._load_trainer_state(self.output_dir)
    
    def _run_with_deepspeed(
        self, 
        zero_stage: DeepSpeedZeroStage,
        *,
        dtype: Optional[TestDtype] = TestDtype.FP32,
        extra_args: Optional[List[str]] = None,
        run_name_note: Optional[str] = None
    ) -> Dict:
        assert isinstance(zero_stage, DeepSpeedZeroStage)
        run_name = f"test_finetuner_deepspeed_{zero_stage.value}"
        if run_name_note:
            run_name += f"_{str(run_name_note)}"
            
        all_args = [
            ["deepspeed"],
            DEEPSPEED_BASE_ARGS,
            [f"{self.lmflow_examples_dir}/finetune.py"],
            DEEPSPEED_ZERO_CONFIG_MAPPING[zero_stage],
            LMFLOW_BASE_ARGS,
        ]
        if dtype == TestDtype.FP32:
            all_args.append(LMFLOW_FP32_ARGS)
        elif dtype == TestDtype.BF16:
            all_args.append(LMFLOW_BF16_ARGS)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        if extra_args and len(extra_args) > 0:
            all_args.append(extra_args)
            
        cmd = self._make_cmd(run_name, all_args)
        res = execute_subprocess_async(cmd)
        self._log_std(res)
        return self._load_trainer_state(self.output_dir)
    
    def _compare_loss(self, trainer_state1: Dict, trainer_state2: Dict):
        loss1 = []
        loss2 = []
        for step_idx in range(TEST_STEPS):
            assert trainer_state1["log_history"][step_idx]["step"] == trainer_state2["log_history"][step_idx]["step"], (
                "log_history step mismatch, check traner_state.json"
            )
            loss1.append(trainer_state1["log_history"][step_idx]["loss"])
            loss2.append(trainer_state2["log_history"][step_idx]["loss"])
        
        self.assertTrue(torch.allclose(torch.tensor(loss1), torch.tensor(loss2), rtol=1e-2, atol=0))
    
    @pytest.mark.lmflow_core
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.skipif(not is_deepspeed_available(), reason="Deepspeed is not available")
    def test_loss_accelerate_dsz3_vs_fsdp(self):        
        dsz3_trainer_state = self._run_with_accelerate(
            backend=AccelerateBackend.DEEPSPEED_ZERO3,
            run_name_note=PeftMethod.NO.value
        )
        fsdp_trainer_state = self._run_with_accelerate(
            backend=AccelerateBackend.FSDP,
            run_name_note=PeftMethod.NO.value
        )
        self._compare_loss(dsz3_trainer_state, fsdp_trainer_state)
        
    @pytest.mark.lmflow_core
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.skipif(not is_deepspeed_available(), reason="Deepspeed is not available")
    def test_loss_deepspeed_z3_vs_accelerate_dsz3(self):
        deepspeed_z3_trainer_state = self._run_with_deepspeed(
            zero_stage=DeepSpeedZeroStage.ZERO3,
            run_name_note=PeftMethod.NO.value
        )
        accelerate_dsz3_train_state = self._run_with_accelerate(
            backend=AccelerateBackend.DEEPSPEED_ZERO3,
            run_name_note=PeftMethod.NO.value
        )
        self._compare_loss(deepspeed_z3_trainer_state, accelerate_dsz3_train_state)
        
        
class TestFinetunerLora(TestFinetunerBase):
    @pytest.mark.lmflow_core
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.skipif(not is_deepspeed_available(), reason="Deepspeed is not available")
    def test_loss_accelerate_dsz3_vs_fsdp(self):
        dsz3_trainer_state = self._run_with_accelerate(
            backend=AccelerateBackend.DEEPSPEED_ZERO3,
            extra_args=LMFLOW_PEFT_ARGS_MAPPING[PeftMethod.LORA],
            run_name_note=PeftMethod.LORA.value
        )
        fsdp_trainer_state = self._run_with_accelerate(
            backend=AccelerateBackend.FSDP,
            extra_args=LMFLOW_PEFT_ARGS_MAPPING[PeftMethod.LORA],
            run_name_note=PeftMethod.LORA.value
        )
        self._compare_loss(dsz3_trainer_state, fsdp_trainer_state)
    
    @pytest.mark.lmflow_core
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.skipif(not is_deepspeed_available(), reason="Deepspeed is not available")
    def test_loss_deepspeed_z3_vs_accelerate_dsz3(self):
        deepspeed_z3_trainer_state = self._run_with_deepspeed(
            zero_stage=DeepSpeedZeroStage.ZERO3,
            extra_args=LMFLOW_PEFT_ARGS_MAPPING[PeftMethod.LORA],
            run_name_note=PeftMethod.LORA.value
        )
        accelerate_dsz3_train_state = self._run_with_accelerate(
            backend=AccelerateBackend.DEEPSPEED_ZERO3,
            extra_args=LMFLOW_PEFT_ARGS_MAPPING[PeftMethod.LORA],
            run_name_note=PeftMethod.LORA.value
        )
        self._compare_loss(deepspeed_z3_trainer_state, accelerate_dsz3_train_state)
        
        
class TestFinetunerQlora(TestFinetunerBase):
    """Currently only supports in bf16"""
    @pytest.mark.lmflow_core
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.skipif(not is_deepspeed_available(), reason="Deepspeed is not available")
    def test_loss_accelerate_dsz3_vs_fsdp(self):
        dsz3_trainer_state = self._run_with_accelerate(
            backend=AccelerateBackend.DEEPSPEED_ZERO3,
            dtype=TestDtype.BF16,
            extra_args=LMFLOW_PEFT_ARGS_MAPPING[PeftMethod.QLORA],
            run_name_note=PeftMethod.QLORA.value
        )
        fsdp_trainer_state = self._run_with_accelerate(
            backend=AccelerateBackend.FSDP,
            dtype=TestDtype.BF16,
            extra_args=LMFLOW_PEFT_ARGS_MAPPING[PeftMethod.QLORA],
            run_name_note=PeftMethod.QLORA.value
        )
        self._compare_loss(dsz3_trainer_state, fsdp_trainer_state)
    
    @pytest.mark.lmflow_core
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.skipif(not is_deepspeed_available(), reason="Deepspeed is not available")
    def test_loss_deepspeed_z3_vs_accelerate_dsz3(self):
        deepspeed_z3_trainer_state = self._run_with_deepspeed(
            zero_stage=DeepSpeedZeroStage.ZERO3,
            dtype=TestDtype.BF16,
            extra_args=LMFLOW_PEFT_ARGS_MAPPING[PeftMethod.QLORA],
            run_name_note=PeftMethod.QLORA.value
        )
        accelerate_dsz3_train_state = self._run_with_accelerate(
            backend=AccelerateBackend.DEEPSPEED_ZERO3,
            dtype=TestDtype.BF16,
            extra_args=LMFLOW_PEFT_ARGS_MAPPING[PeftMethod.QLORA],
            run_name_note=PeftMethod.QLORA.value
        )
        self._compare_loss(deepspeed_z3_trainer_state, accelerate_dsz3_train_state)
               
        
class TestFinetunerCustomOptim(TestFinetunerBase):
    @pytest.mark.lmflow_core
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.skipif(not is_deepspeed_available(), reason="Deepspeed is not available")
    def test_loss_accelerate_dsz3_vs_fsdp(self):
        dsz3_trainer_state = self._run_with_accelerate(
            backend=AccelerateBackend.DEEPSPEED_ZERO3,
            extra_args=LMFLOW_CUSTOM_OPTIM_ARGS,
            run_name_note="custom_optim"
        )
        fsdp_trainer_state = self._run_with_accelerate(
            backend=AccelerateBackend.FSDP,
            extra_args=LMFLOW_CUSTOM_OPTIM_ARGS,
            run_name_note="custom_optim"
        )
        self._compare_loss(dsz3_trainer_state, fsdp_trainer_state)
    
    @pytest.mark.lmflow_core
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.skipif(not is_deepspeed_available(), reason="Deepspeed is not available")
    def test_loss_deepspeed_z3_vs_accelerate_dsz3(self):
        deepspeed_z3_trainer_state = self._run_with_deepspeed(
            zero_stage=DeepSpeedZeroStage.ZERO3,
            extra_args=LMFLOW_CUSTOM_OPTIM_ARGS,
            run_name_note="custom_optim"
        )
        accelerate_dsz3_train_state = self._run_with_accelerate(
            backend=AccelerateBackend.DEEPSPEED_ZERO3,
            extra_args=LMFLOW_CUSTOM_OPTIM_ARGS,
            run_name_note="custom_optim"
        )
        self._compare_loss(deepspeed_z3_trainer_state, accelerate_dsz3_train_state)