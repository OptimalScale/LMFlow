"""The Evaluator class simplifies the process of running evaluation on a language model provided by a HFDecoderModel instance imported from the lmflow package. The class constructor takes three dictionaries as arguments: model_args containing arguments related to the language model, data_args containing arguments related to the data used for evaluation, and evaluator_args containing other arguments for the evaluation process.

The class has two methods: create_dataloader() that loads the data from the test file, creates a data loader, and returns it with the size of the data, and evaluate(model) that generates output text given input text. It uses the create_dataloader() method to load the data, iterates over the data in mini-batches, and encodes the input text with the encode() method of the HFDecoderModel class. Then, it generates output text using the evaluate() method of the HFDecoderModel class, decodes the generated output text using the decode() method of the HFDecoderModel class, and writes the output to a file in the output directory. The method also logs some information to the console and Weights and Biases if the use_wandb argument is True.
"""
import os
import torch
import wandb
import deepspeed
import sys
import numpy as np
import datetime
import json
# TODO: remove later
from accelerate import Accelerator
from transformers import AutoConfig
import torch.distributed as dist

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.base_pipeline import BasePipeline
from lmflow.models.hf_decoder_model import HFDecoderModel
from lmflow.utils.data_utils import set_random_seed, batchlize, answer_extraction
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

class Evaluator(BasePipeline):
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

        # logger
        if(self.evaluator_args.use_wandb == True):
            wandb.init(project="lmflow_evaluation")
        # random seed
        set_random_seed(self.evaluator_args.random_seed)
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        torch.cuda.set_device(self.local_rank)  # NOTE: cpu-only machine will have error

        if evaluator_args.use_accelerator_for_evaluator:
            self.accelerator = Accelerator()
            self.accelerator.wait_for_everyone()
        else:
            deepspeed.init_distributed()

        self.config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        try: 
            self.model_hidden_size = self.config.hidden_size
        except:
            print("Error in setting hidden size, use the default size 1024")
            self.model_hidden_size = 1024 # gpt2 seems do not have hidden_size in config

        print(f"model_hidden_size = {self.model_hidden_size}")
        # batch size has to be divisible by world_size, but can be bigger than world_size
        train_batch_size = self.evaluator_args.inference_batch_size_per_device * self.world_size
        self.evaluator_args.minibatch_size = train_batch_size
        self.block_size = evaluator_args.evaluate_block_size
        # dataloader, data_size = create_dataloader(args)    # load dataset


    def create_dataloader(self, dataset: Dataset):
        data_dict = dataset.to_dict()
        inputs = [ instance["input"] for instance in data_dict["instances"] ]
        outputs = [ instance["output"] for instance in data_dict["instances"] ]
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
            self.evaluator_args.minibatch_size,
            self.evaluator_args.random_shuffle
        )
        print(f"Successfully create dataloader with size {len(dataloader)},batch_size {self.evaluator_args.minibatch_size}.")
        
        return dataloader, dataset_size


    # TODO: Split for better unittest

    def _match(self, predicted_answer, groundtruth, answer_type=None):
        case_insensitive_types = [
            "strategyqa",
            "coin_flip",
            "pubmedqa",
            "binary_choice",
            "medmcqa",
            "usmle",
        ]
        if answer_type in case_insensitive_types:
            return predicted_answer.lower() == groundtruth.lower()
        else:
            return predicted_answer == groundtruth
        return False


    def evaluate(
        self,
        model,
        dataset: Dataset,
        metric = "accuracy",
        verbose=True,
    ):
        """
        Perform Evaluation for a model

        Parameters
        ------------
        model : TunableModel object.
            TunableModel to perform inference

        dataset : Dataset object.
            

        """
        if metric in ["acc", "accuracy"]:
            if self.evaluator_args.use_accelerator_for_evaluator:
                acc = self._evaluate_acc_with_accelerator(model, dataset, verbose=verbose)
            else:
                acc = self._evaluate_acc_with_deepspeed(model, dataset, verbose=verbose)
            print(f"Evaluating final accuracy: {acc}")
            return acc
        elif metric in ["ppl", "perplexity"]:
            ppl = self._evaluate_ppl(model, dataset, verbose=verbose)
            print(f"Evaluating final perplexity: {ppl}")
            return ppl
        elif metric in ["nll", "neg_log_likelihood"]:
            nll = self._evaluate_nll(model, dataset, verbose=verbose)
            print(f"Evaluating final negative log likelihood: {nll}")
            return nll
        else:
            raise NotImplementedError(f"metric {metric} is not supported")


    def _evaluate_acc_with_accelerator(self, model, dataset, verbose=True):
        dataloader, data_size = self.create_dataloader(dataset)
        if self.accelerator.is_local_main_process:
            if not os.path.exists(self.evaluator_args.output_dir):
                os.makedirs(self.evaluator_args.output_dir)
            output_writer = open(f"{self.evaluator_args.output_dir}/evaluation.json", "w")
        
        correct_number_list = []
        for batch_index, batch in enumerate(dataloader):
            if batch_index * self.world_size >= self.data_args.max_eval_samples: 
                break
            if self.local_rank*self.evaluator_args.inference_batch_size_per_device >= len(batch):
                current_batch = batch[:self.evaluator_args.inference_batch_size_per_device]
            else:
                current_batch = batch[self.local_rank*self.evaluator_args.inference_batch_size_per_device:(self.local_rank+1)*self.evaluator_args.inference_batch_size_per_device]
            prompt_structure = self.evaluator_args.prompt_structure
            input = [prompt_structure.format(input=i['input']) for i in current_batch]
            output = [i['output'] for i in current_batch]   

            batch_input = model.encode(input, return_tensors="pt",padding=True).to(self.accelerator.device)
            inputs = batch_input['input_ids']
            mask = batch_input['attention_mask']
            with self.accelerator.autocast():
                outputs = model.inference(inputs, max_new_tokens=self.evaluator_args.max_new_tokens,attention_mask=mask,temperature=self.evaluator_args.temperature, repetition_penalty=self.evaluator_args.repetition_penalty,use_accelerator=self.evaluator_args.use_accelerator_for_evaluator)
            text_out = model.decode(outputs, skip_special_tokens=True)
            decoded_input = model.decode(inputs, skip_special_tokens=True,)
            prompt_length = [len(i) for i in decoded_input]
            text_out = [text_out[i][prompt_length[i]:] for i in range(len(text_out))]
            answer_type = self.evaluator_args.answer_type
            pred_answer = []
            for i in text_out:
                pred_answer.append(answer_extraction(
                    i,
                    answer_type=answer_type,
                ))
            if verbose:
                print(f"batch_index{batch_index} rank{self.local_rank}:\n   question={input}\n  prediction={text_out}\n")
                print(f"predicted answer: {pred_answer} \n")
                print(f"groundtruth answer: {output} \n")

            if self.local_rank * self.evaluator_args.inference_batch_size_per_device  >= len(batch):
                correct_ = 0
            else:
                correct_ = 0
                for i in range(len(pred_answer)):
                    if self._match(pred_answer[i], output[i], answer_type):
                        correct_ += 1

            # collect accuracy from all gpus
            all_process = torch.tensor([correct_], dtype=torch.float32, device=self.local_rank)
            all_process = self.accelerator.gather(all_process)
            correct_ = sum(all_process.tolist())
            correct_number_list.append(correct_)

            # collect predictions from all gpus
            output_dict = {"question": input,
                        "prediction": text_out,
                        "pred_answer": pred_answer,
                        "answer": output}
            if(self.world_size > 1):
                all_process_list = [{}] * self.world_size
                dist.gather_object(output_dict, all_process_list if dist.get_rank() == 0 else None, dst=0)
            else:
                all_process_list = [output_dict]
            
            if self.accelerator.is_local_main_process:
                current_total = (batch_index+1) * self.world_size * self.evaluator_args.inference_batch_size_per_device
                current_accuracy = np.sum(correct_number_list) / current_total if int(current_total) < data_size else np.sum(correct_number_list) / data_size
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{int(current_total) if int(current_total) < data_size else data_size} / {data_size} has been finished, # correct = { np.sum(correct_number_list)}, current accuracy = {current_accuracy}")

                if(self.evaluator_args.use_wandb == True):
                    wandb.log({"Accuracy": current_accuracy})

                for index, output in enumerate(all_process_list):
                    output_json = json.dumps(output)
                    output_writer.write(output_json + '\n')

        if self.accelerator.is_local_main_process:
            current_accuracy = np.sum(correct_number_list) / data_size
            print(f"# Correct = {np.sum(correct_number_list)}, # Total = {data_size}, Final accuracy = ", current_accuracy)
            output_writer.close()
        return np.sum(correct_number_list) / data_size

    def _evaluate_acc_with_deepspeed(self, model, dataset, verbose=True):
        dataloader, data_size = self.create_dataloader(dataset)
        if not dist.is_initialized() or dist.get_rank() == 0:
            if not os.path.exists(self.evaluator_args.output_dir):
                os.makedirs(self.evaluator_args.output_dir)
            output_writer = open(f"{self.evaluator_args.output_dir}/evaluation.json", "w")

        correct_number_list = []
        for batch_index, batch in enumerate(dataloader):
            if batch_index * self.world_size >= self.data_args.max_eval_samples: 
                break
            if self.local_rank*self.evaluator_args.inference_batch_size_per_device >= len(batch):
                current_batch = batch[:self.evaluator_args.inference_batch_size_per_device]
            else:
                current_batch = batch[self.local_rank*self.evaluator_args.inference_batch_size_per_device:(self.local_rank+1)*self.evaluator_args.inference_batch_size_per_device]
            prompt_structure = self.evaluator_args.prompt_structure
            input = [prompt_structure.format(input=i['input']) for i in current_batch]
            output = [i['output'] for i in current_batch]   
            input_idx = [i['input_idx'] for i in current_batch]
            batch_input = model.encode(input, return_tensors="pt",padding=True).to(device=self.local_rank)
            inputs = batch_input['input_ids']
            mask = batch_input['attention_mask']
            outputs = model.inference(inputs, max_new_tokens=self.evaluator_args.max_new_tokens, attention_mask=mask,temperature=self.evaluator_args.temperature, repetition_penalty=self.evaluator_args.repetition_penalty)
            text_out = model.decode(outputs, skip_special_tokens=True)
            # # only return the generation, trucating the input
            decoded_input = model.decode(inputs, skip_special_tokens=True,)
            prompt_length = [len(i) for i in decoded_input]
            text_out = [text_out[i][prompt_length[i]:] for i in range(len(text_out))]
            answer_type = self.evaluator_args.answer_type
            pred_answer = []
            for i in text_out:
                pred_answer.append(answer_extraction(
                    i,
                    answer_type=answer_type,
                ))
            if verbose:
                print(f"batch_index{batch_index} rank{self.local_rank}:\n   question={input}\n  prediction={text_out}\n")
                print(f"predicted answer: {pred_answer} \n")
                print(f"groundtruth answer: {output} \n")

            if self.local_rank * self.evaluator_args.inference_batch_size_per_device  >= len(batch):
                correct_ = 0
            else:
                correct_ = 0
                for i in range(len(pred_answer)):
                    if self._match(pred_answer[i], output[i], answer_type):
                        correct_ += 1

            # collect accuracy from all gpus
            all_process = torch.tensor([correct_], dtype=torch.float32, device=self.local_rank)
            dist.all_reduce(all_process, dist.ReduceOp.SUM, async_op=False)
            correct_ = all_process.tolist()
            correct_number_list.append(correct_)

            # collect predictions from all gpus
            output_dict = {"question": input,
                        "prediction": text_out,
                        "pred_answer": pred_answer,
                        "answer": output}
            all_process_list = [{}] * self.world_size

            dist.gather_object(output_dict, all_process_list if dist.get_rank() == 0 else None, dst=0)
            if not dist.is_initialized() or dist.get_rank() == 0:
                current_total = (batch_index+1) * self.world_size * self.evaluator_args.inference_batch_size_per_device
                current_accuracy = np.sum(correct_number_list) / current_total if int(current_total) < data_size else np.sum(correct_number_list) / data_size
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{int(current_total) if int(current_total) < data_size else data_size} / {data_size} has been finished, # correct = { np.sum(correct_number_list)}, current accuracy = {current_accuracy}")

                if(self.evaluator_args.use_wandb == True):
                    wandb.log({"Accuracy": current_accuracy})

                for index, output in enumerate(all_process_list):
                    output_json = json.dumps(output)
                    output_writer.write(output_json + '\n')

        if not dist.is_initialized() or dist.get_rank() == 0:
            current_accuracy = np.sum(correct_number_list) / data_size
            print(f"# Correct = {np.sum(correct_number_list)}, # Total = {data_size}, Final accuracy = ", current_accuracy)
            output_writer.close()
        return np.sum(correct_number_list) / data_size

    def _evaluate_ppl(self, model, dataset: Dataset, verbose=True):
        data_dict = dataset.to_dict()
        if data_dict['type'] == 'text2text':
            raise NotImplementedError("ppl evaluation is currently not supported for text2text dataset, please use text_only dataset.")
        texts = [ instance["text"] for instance in data_dict["instances"] ]
        encodings = model.get_tokenizer()("\n\n".join(texts), return_tensors="pt")
        # Define some constant
        try:
            max_length = min(model.get_backend_model().config.n_positions, model.get_max_length())
        except:
            max_length = min(1024, model.get_max_length())

        if verbose:
            print(f"The maximum sequence length : {max_length}")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.block_size):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from block_size on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device=self.local_rank)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model.get_backend_model()(input_ids, labels=target_ids)
                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if verbose:
                print(f"Evaluating PPL: {int(begin_loc/self.block_size) + 1} / {int(seq_len/self.block_size)} Complete, current ppl : {torch.exp(torch.stack(nlls).mean())}")
            if end_loc == seq_len:
                break
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl


    def _evaluate_nll(
        self,
        model,
        dataset: Dataset,
        verbose=True,
    ):
        """
        Evaluates negative log likelihood of the model over a dataset.

        NLL = -1/N sum_{i=1}^N sum_{j=1}^|w_i| ln(p(w_{i,j}|context_window)),

        where N is the number of data samples, w_{i,j} is the j-th token in
        i-th sample. Here "context_window" = p(w_{i,start}, w_{i,start+1}, ...,
        p_{i,j-1} with start = max(0, j - window_length + 1). "window_length"
        is normally the maximum length accepted by the model.

        Returns:
            A float which represents the negative log likelihood.
        """
        data_dict = dataset.to_dict()

        # Handles prompt structure
        if dataset.get_type() == "text2text":
            prompt = self.evaluator_args.prompt_structure
            data_dict["instances"] = [
                {
                    "input": prompt.format(input=instance["input"]),
                    "output": instance["output"]
                }
                for instance in data_dict["instances"]
            ]

        dataset = dataset.from_dict(data_dict)
        tokenized_dataset = model.tokenize(dataset, add_special_tokens=False)
        tokenized_dataset = tokenized_dataset.get_backend_dataset()
        encoding_list = [
            {
                "input_ids": torch.tensor([input_ids]),
                "labels": torch.tensor([labels]),
            }
            for input_ids, labels in zip(tokenized_dataset["input_ids"],
                                         tokenized_dataset["labels"])
        ]

        # Gets context window length
        try:
            max_length = min(model.get_backend_model().config.n_positions,
                             model.get_max_length())
        except:
            max_length = min(1024, model.get_max_length())

        nlls = []
        full_nlls = []
        num_samples = len(encoding_list)
        for sample_idx, encodings in enumerate(encoding_list):
            seq_len = encodings["input_ids"].size(1)

            prev_end_loc = 0
            for begin_loc in range(0, seq_len, self.block_size):
                end_loc = min(begin_loc + max_length, seq_len)

                # may be different from block_size on last loop
                trg_len = end_loc - prev_end_loc
                input_ids = encodings["input_ids"][:, begin_loc:end_loc]
                input_ids = input_ids.to(device=self.local_rank)

                labels = encodings["labels"][:, begin_loc:end_loc]
                target_ids = labels.clone()
                full_target_ids = input_ids.clone()

                def get_nll(label_ids, nll_list):
                    label_ids[:, :-trg_len] = -100
                    label_ids = label_ids.to(device=self.local_rank)

                    # Valid labels are from 0 to `vocab_size`
                    num_valid_labels = torch.count_nonzero(label_ids >= 0)
                    if label_ids[0, 0] != -100:
                        num_valid_labels -= 1

                    if not torch.all(label_ids == -100):
                        with torch.no_grad():
                            outputs = model.get_backend_model()(
                                input_ids, labels=label_ids
                            )
                            # loss is calculated using CrossEntropyLoss which
                            # sums over valid labels N.B. the model only
                            # calculates loss over trg_len - 1 labels, because
                            # it internally shifts the labels to the left by 1.
                            neg_log_likelihood = outputs.loss * num_valid_labels
                    else:
                        neg_log_likelihood = torch.zeros([]).to(
                            device=self.local_rank
                        )

                    nll_list.append(neg_log_likelihood)

                get_nll(target_ids, nlls)
                get_nll(full_target_ids, full_nlls)

                current_output_nll = torch.stack(nlls).sum() / (sample_idx + 1)
                current_full_nll = torch.stack(full_nlls).sum() / (sample_idx + 1)

                prev_end_loc = end_loc
                if verbose:
                    if dataset.get_type() == "text_only":
                        print(
                            f"Evaluating negative log likelihood:"
                            f" {sample_idx + 1} / {num_samples} Complete,"
                            f" current nll: {current_full_nll}"
                        )
                    elif dataset.get_type() == "text2text":
                        print(
                            f"Evaluating negative log likelihood:"
                            f" {sample_idx + 1} / {num_samples} Complete,"
                            f" current full nll / input nll / output nll:"
                            f" {current_full_nll} /"
                            f" {current_full_nll - current_output_nll} /"
                            f" {current_output_nll}"
                        )
                    else:
                        raise NotImplementedError(
                            "f{dataset.get_type()} typed datasets are not"
                            " supported"
                        )

                if end_loc == seq_len:
                    break

        mean_nll = torch.stack(nlls).sum() / num_samples
        return mean_nll
