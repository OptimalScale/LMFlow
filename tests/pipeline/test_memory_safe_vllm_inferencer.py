import unittest
import json

from lmflow.args import DatasetArguments, ModelArguments, InferencerArguments
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.pipeline.inferencerv2 import MemorySafeVLLMInferencer
from lmflow.datasets import Dataset


model_args = ModelArguments(
    '/home/yizhenjia/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/ff3a49fac17555b8dfc4db6709f480cc8f16a9fe', 
    torch_dtype='auto',
    vllm_gpu_memory_utilization=0.95,
    use_vllm_inference=True,
    vllm_tensor_parallel_size=2,
)
data_args = DatasetArguments(
    '/vol/yizhenjia/projs/LMFlow/data/alpaca/test_conversation',
    preprocessing_num_workers=4,
)
inferencer_args = InferencerArguments(
    random_seed=42,
    apply_chat_template=True,
    num_output_sequences=2,
    temperature=1.0,
    max_new_tokens=1024,
    memory_safe_vllm_inference_devices='0,1,', # intended, testing InferencerArguments post init.
    save_results=True,
    results_path='/vol/yizhenjia/projs/LMFlow/data/mem_safe_vllm_res.json',
    memory_safe_vllm_inference_detokenize=False
)


class MemorySafeVLLMInferencerTest(unittest.TestCase):
    def test_init(self):
        self.dataset = Dataset(data_args)
        self.model = HFDecoderModel(model_args)
        self.inferencer = MemorySafeVLLMInferencer(
            model_args=model_args,
            data_args=data_args,
            inferencer_args=inferencer_args,
        )

    def test_inference(self):
        res = self.inferencer.inference()
        self.assertTrue(isinstance(res, list))
        self.assertTrue(isinstance(res[0], list))
        self.assertTrue(isinstance(res[0][0], list))
        self.assertTrue(isinstance(res[0][0][0], int))
        
    def test_inference_detokenize(self):
        inferencer_args.memory_safe_vllm_inference_detokenize = True
        self.inferencer = MemorySafeVLLMInferencer(
            model_args=model_args,
            data_args=data_args,
            inferencer_args=inferencer_args,
        )
        res = self.inferencer.inference()
        self.assertTrue(isinstance(res, list))
        self.assertTrue(isinstance(res[0], list))
        self.assertTrue(isinstance(res[0][0], str))
        
        
if __name__ == "__main__":
    unittest.main()