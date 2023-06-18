import os
import deepspeed
import torch
import wandb
import sys
import numpy as np
import datetime
import json
from rouge_score import rouge_scorer
from multiprocessing import Pool
from functools import partial
# TODO: remove later
from transformers import AutoConfig
# from lmflow.pipeline.auto_pipeline import AutoPipeline
import evaluate
import torch.distributed as dist
from transformers import HfArgumentParser
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.base_pipeline import BasePipeline
from lmflow.utils.data_utils import set_random_seed, batchlize, answer_extraction, load_data
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

# copied from evaluate.py, with small changes.
pipeline_name = "test_rougel"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

with open (pipeline_args.deepspeed, "r") as f:
    ds_config = json.load(f)

model = AutoModel.get_model(model_args, tune_strategy='none', ds_config=ds_config)
dataset = Dataset(data_args)

evaluator = AutoPipeline.get_pipeline(
    pipeline_name=pipeline_name,
    model_args=model_args,
    data_args=data_args,
    pipeline_args=pipeline_args,
)
evaluator.evaluate(model=model, dataset=dataset, metric=pipeline_args.metric)

#copied from evaluator.py
class Test_rougel(BasePipeline):
    """
    Initializes the `Evaluator` class with given arguments.

    Parameters
    ------------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.

    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    evaluator_args : EvaluatorArguments object.
        Contains the arguments required to perform evaluation.


    """

    def __init__(self, model_args, data_args, evaluator_args):
        # our method
        self.data_args = data_args
        self.evaluator_args = evaluator_args
        self.model_args = model_args
        print("--------Begin Evaluator Arguments----------")
        print(f"model_args : {self.model_args}")
        print(f"data_args : {self.data_args}")
        print(f"evaluator_args : {self.evaluator_args}")
        print("--------End Evaluator Arguments----------")
        # logger
        if (self.evaluator_args.use_wandb == True):
            wandb.init(project="lmflow_evaluation")
        # random seed
        set_random_seed(self.evaluator_args.random_seed)
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        print("\nself.world_size是：", self.world_size, "\n")
        torch.cuda.set_device(self.local_rank)  # NOTE: cpu-only machine will have error
        deepspeed.init_distributed()

        self.config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        try:
            self.model_hidden_size = self.config.hidden_size
        except:
            print("Error in setting hidden size, use the default size 1024")
            self.model_hidden_size = 1024  # gpt2 seems do not have hidden_size in config

        print(f"model_hidden_size = {self.model_hidden_size}")
        # batch size has to be divisible by world_size, but can be bigger than world_size
        train_batch_size = 1 * self.world_size
        self.evaluator_args.minibatch_size = train_batch_size
        self.block_size = evaluator_args.evaluate_block_size
        # dataloader, data_size = create_dataloader(args)    # load dataset

    # First use the method in self-instruct to get the ROUGE-L scores for the dataset, then use the method in LMFlow and compare the two scores,
    # The metric is tested to be valid if all scores are the same.
    def get_rougel_score_list(self, predicted_data: str):
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

        dataset = load_data(predicted_data)
        data_dict = dataset.to_dict()
        inputs = [instance["input"] for instance in data_dict["instances"]]
        outputs = [instance["output"] for instance in data_dict["instances"]]
        dataset_size = len(outputs)

        dataset_buf = []
        for idx in range(dataset_size):
            dataset_buf.append({
                "input": inputs[idx],
                "output": outputs[idx],
                "input_idx": idx
            })

        dataloader = batchlize(   # 相当于每minibatch_size大小切一段，dataloader = [[{}, {}, ... ], [{}, {}, ... ], ... ]
            dataset_buf,
            self.evaluator_args.minibatch_size,  # = self.world_size
            self.evaluator_args.random_shuffle
        )
        print(f"Successfully create dataloader with size {len(dataloader)}.")

        score_list = []  # store the maximum ROUGE-L score in each batch

        for batch in dataloader:
            input_ = [data["input"] for data in batch]
            output_ = [data["output"] for data in batch]
            with Pool(4) as p:  # 4 processes
                rouge_scores = p.map(partial(scorer.score, input_), [output_])
            rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]  # score["rougeL"].fmeasure 是对应的pair的得分
            max_rl_score = max(rouge_scores)
            score_list.append(max_rl_score)

        return score_list


    def create_dataloader(self, dataset: Dataset):
        data_dict = dataset.to_dict()
        inputs = [instance["input"] for instance in data_dict["instances"]]
        outputs = [instance["output"] for instance in data_dict["instances"]]
        dataset_size = len(outputs)
        dataset_buf = []
        for idx in range(dataset_size):
            dataset_buf.append({
                "input": inputs[idx],
                "output": outputs[idx],
                "input_idx": idx
            })

        dataloader = batchlize(
            dataset_buf,
            self.evaluator_args.minibatch_size,  # = self.world_size
            self.evaluator_args.random_shuffle
        )
        print(f"Successfully create dataloader with size {len(dataloader)}.")
        return dataloader, dataset_size

    # TODO: Split for better unittest
    def _calculate_rouge_l(self, predicted_answer, groundtruth, scorer: rouge_scorer.RougeScorer, answer_type=None):
        case_insensitive_types = [
            "strategyqa",
            "coin_flip",
            "pubmedqa",
            "binary_choice",
            "medmcqa",
            "usmle",
        ]
        if answer_type in case_insensitive_types:
            rouge_score = scorer.score(groundtruth.lower(), predicted_answer.lower())["rougeL"].fmeasure
        else:
            rouge_score = scorer.score(groundtruth, predicted_answer)["rougeL"].fmeasure
        return rouge_score

    def evaluate(self, model, dataset: Dataset, metric="rougel"):
        """
        Perform Evaluation for a model

        Parameters
        ------------
        model : TunableModel object.
            TunableModel to perform inference

        dataset : Dataset object.


        """
        if metric in ["rl", "rouge-l", "ROUGE-L"]:
            dataloader, data_size = self.create_dataloader(dataset)  # data_size = number of mini-batches

            if not dist.is_initialized() or dist.get_rank() == 0:
                if not os.path.exists(self.evaluator_args.output_dir):
                    os.makedirs(self.evaluator_args.output_dir)
                output_writer = open(f"{self.evaluator_args.output_dir}/evaluation.json", "w")

            pred_score_list = []  # list to record the ROUGE-L scores of all batches from LMFlow method

            # ds_engine = deepspeed.initialize(model=model.get_model(), config_params=self.ds_config)[0]
            # ds_engine.module.eval()
            for batch_index, batch in enumerate(dataloader):
                if batch_index * self.world_size >= self.data_args.max_eval_samples:
                    break
                if self.local_rank >= len(batch):
                    current_batch = batch[0]
                else:
                    # the batch in current process
                    current_batch = batch[self.local_rank]

                prompt_structure = self.evaluator_args.prompt_structure
                input = prompt_structure.format(input=current_batch['input'])
                output = current_batch['output']
                input_idx = current_batch['input_idx']

                inputs = model.encode(input, return_tensors="pt").to(device=self.local_rank)

                # with torch.no_grad():
                # outputs = ds_engine.module.generate(inputs, synced_gpus=True, pad_token_id=model.get_tokenizer().eos_token_id, min_length=5, max_length=100,temperature=0.0, do_sample=False)
                outputs = model.inference(inputs, max_new_tokens=100, temperature=0.0)
                text_out = model.decode(outputs[0], skip_special_tokens=True)

                # # only return the generation, truncating the input
                prompt_length = len(model.decode(inputs[0], skip_special_tokens=True, ))
                text_out = text_out[prompt_length:]
                answer_type = self.evaluator_args.answer_type
                pred_answer = answer_extraction(
                    text_out,
                    answer_type=answer_type,
                )
                print(
                    f"batch_index{batch_index} rank{self.local_rank}:\n   question={input}\n  prediction={text_out}\n")
                print(f"predicted answer: {pred_answer} \n")
                print(f"groundtruth answer: {output} \n")


                scorer = rouge_scorer.RougeScorer(["rougeL"],
                                                  use_stemmer=False)  # stemmer: stem the words to their root form

                if self.local_rank >= len(
                        batch):  # for last batch, the padding examples are ignored and do not contribute to the ROUGE-L
                    rl_ = 0
                    total_ = 0
                else:
                    rl_ = max(0, self._calculate_rouge_l(pred_answer, output, scorer, answer_type))
                    total_ = 1
                score = rl_

                # collect rouge-l from all gpus
                all_process = torch.tensor([rl_, total_], dtype=torch.float32, device=self.local_rank)
                dist.all_reduce(all_process, dist.ReduceOp.MAX, async_op=False)
                max_, total_ = all_process.tolist()
                print("max_: ", max_)
                print("total_: ", total_)
                # avg = max_ / total_
                avg = max_
                pred_score_list.append(avg)
                # total += total_

                # collect predictions from all gpus
                output_dict = {"question": input,
                               "prediction": text_out,
                               "pred_answer": pred_answer,
                               "answer": output}
                all_process_list = [{}] * self.world_size

                dist.gather_object(output_dict, all_process_list if dist.get_rank() == 0 else None,
                                   dst=0)
                print("all_process_list: ", all_process_list)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    current_rouge_l = np.mean(pred_score_list)
                    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          "{}/ {} has been finished, current ROUGE-L = {}".format(int(pred_score_list), data_size,
                                                                                  current_rouge_l))

                    if (self.evaluator_args.use_wandb == True):
                        wandb.log({"rougeL_fmeasure": current_rouge_l})

                    for index, output in enumerate(all_process_list):
                        output_json = json.dumps(output)
                        output_writer.write(output_json + '\n')

            if not dist.is_initialized() or dist.get_rank() == 0:
                current_rouge_l = np.mean(pred_score_list)
                print("Final ROUGE-L = ", current_rouge_l)
                output_writer.close()

        else:
            raise NotImplementedError(f"{metric} is not implemented or not match with our defined metrics")

        # load the dataset with predicted answers and apply the self-instruct method to get the answer score list.
        ans_score_list = self.get_rougel_score_list(f"{self.evaluator_args.output_dir}/evaluation.json")

        # Start compare the two ROUGE-L scores lists we get
        matched = True
        for pred, ans in zip(pred_score_list, ans_score_list):
            print("LMFlow ROUGE-L: ", pred, " -- self-instruct ROUGE-L: ", ans)
            if pred != ans:
                matched = False
                print("scores not matched!")
                return
        print("scores matched. Tested to be valid.")



