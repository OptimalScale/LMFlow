# cannot use unittest, since memory safe vllm inference uses stdout, 
# which has conflicts with unittest stdout.
import logging
import json

from lmflow.args import DatasetArguments, ModelArguments, InferencerArguments
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.pipeline.vllm_inferencer import MemorySafeVLLMInferencer
from lmflow.datasets import Dataset


logger = logging.getLogger(__name__)


model_args = ModelArguments(
    'Qwen/Qwen2-0.5B', 
    torch_dtype='auto',
)
data_args = DatasetArguments(
    './data/alpaca/test_conversation',
    preprocessing_num_workers=4,
)
inferencer_args = InferencerArguments(
    random_seed=42,
    apply_chat_template=True,
    num_output_sequences=2,
    temperature=1.0,
    max_new_tokens=1024,
    save_results=True,
    results_path='./data/mem_safe_vllm_res.json',
    use_vllm=True,
    enable_decode_inference_result=False,
    vllm_gpu_memory_utilization=0.95,
    vllm_tensor_parallel_size=2,
)


class MemorySafeVLLMInferencerTest:
    def test_init(self):
        self.dataset = Dataset(data_args)
        self.model = HFDecoderModel(model_args)
        self.inferencer = MemorySafeVLLMInferencer(
            model_args=model_args,
            data_args=data_args,
            inferencer_args=inferencer_args,
        )
        self.status = []

    def test_inference(self):
        res = self.inferencer.inference()
        test_res = all([
            isinstance(res, list), 
            isinstance(res[0], list), 
            isinstance(res[0][0], list), 
            isinstance(res[0][0][0], int),
        ])
        self.status.append(test_res)
        logger.warning(f"test_inference: {test_res}")
        
    def test_inference_detokenize(self):
        inferencer_args.enable_decode_inference_result = True
        self.inferencer = MemorySafeVLLMInferencer(
            model_args=model_args,
            data_args=data_args,
            inferencer_args=inferencer_args,
        )
        res = self.inferencer.inference()
        test_res = all([
            isinstance(res, list), 
            isinstance(res[0], list), 
            isinstance(res[0][0], str), 
        ])
        self.status.append(test_res)
        logger.warning(f"test_inference_detokenize: {test_res}")
        
    def summary(self):
        logger.warning(f"MemorySafeVLLMInferencerTest: {all(self.status)}")
        
        
if __name__ == "__main__":
    test = MemorySafeVLLMInferencerTest()
    test.test_init()
    test.test_inference()
    test.test_inference_detokenize()
    test.summary()