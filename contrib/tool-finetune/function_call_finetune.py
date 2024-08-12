import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
import hashlib
from typing import Dict, List, Union, Tuple, Optional, Sequence
import logging
from datasets import Features
import transformers
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.testing_utils import CaptureLogger
from transformers import HfArgumentParser, PreTrainedTokenizer, PreTrainedTokenizerFast
from peft import LoraConfig, TaskType
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.tokenization.hf_decoder_model import blocking
from lmflow.utils.conversation_template.base import TemplateComponent
from lmflow.utils.constants import (
    TEXT_ONLY_DATASET_DESCRIPTION,
    TEXT2TEXT_DATASET_DESCRIPTION,
    CONVERSATION_DATASET_DESCRIPTION,
    CONVERSATION_ROLE_NAMES
)
from lmflow.utils.conversation_template import ConversationTemplateForTool, PRESET_TEMPLATES
from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)
from lmflow.models.hf_decoder_model import HFDecoderModel
tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
logger = logging.getLogger(__name__)


class HFDecoderModelForTool(HFDecoderModel):
    def tokenize(
        self, 
        dataset, 
        add_special_tokens=True, 
        *args, 
        **kwargs
    ) -> Dataset:
        """
        Tokenize the full dataset.
    
        Parameters
        ------------
        dataset : lmflow.datasets.Dataset.

        args : Optional.
            Positional arguments.
        
        kwargs : Optional.
            Keyword arguments.    
        
        Returns
        ------------
        tokenized_datasets :
            The tokenized dataset, without any leading or trailing special
            tokens (normally they are Begin-Of-Sentence or End-Of-Sentence
            tokens).
        """
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if dataset.get_backend() != "huggingface":
            raise NotImplementedError(
                "tokenization of datasets with non-huggingface backend are"
                "not supported yet"
            )

        dataset_type = dataset.get_type()
        model_args = self.model_args
        raw_datasets = dataset
        hf_raw_datasets = dataset.get_backend_dataset()
        column_names = list(hf_raw_datasets.features)
        data_args = raw_datasets.get_data_args()

        # Requires three types of information for tokenizing different datasets
        #   1) Which fields require tokenization, e.g.
        #        "text2float": "text", but not "float"
        #        "text2text": both "input" and "output"
        #   2) How will there tokenized sequence concatenated together, e.g.
        #        "text_only": "text" -> "text"
        #        "text2text": "input", "output" -> "input" + "output"
        #   3) Which fields require loss in final computation, e.g.
        #        "text_only": "text"
        #        "text2text": "output" only
        tokenized_column_order = None       # Handles 1) and 2)
        label_columns = None                # Handles 3)
        if dataset_type == "text_only":
            tokenized_column_order = ["text"]
            label_columns = ["text"]
        elif dataset_type == "text2text":
            tokenized_column_order = ["input", "output"]
            label_columns = ["output"]
            add_special_tokens = False
        elif dataset_type == "conversation":
            if data_args.conversation_template:
                if data_args.conversation_template in PRESET_TEMPLATES.keys():
                    conversation_template = PRESET_TEMPLATES[data_args.conversation_template]
                else:
                    raise NotImplementedError(
                        f"Conversation template {data_args.conversation_template} is not supported yet."
                    )
            else:
                logger.warning("No conversation template provided. Using default template.")
                conversation_template = PRESET_TEMPLATES['empty']
                        
            logger.warning(f"Conversation template: {conversation_template}")
        else:
            raise NotImplementedError(
                f"dataset type \"{dataset_type}\" is not supported, currently"
                " only support following data types:\n"
                f"    1) {TEXT_ONLY_DATASET_DESCRIPTION}\n"
                f"    2) {TEXT2TEXT_DATASET_DESCRIPTION}\n"
                f"    3) {CONVERSATION_DATASET_DESCRIPTION}\n"
            )

        # Whether to truncate long sequences to fit into max_length
        use_truncation = False
        if model_args.use_lora or data_args.disable_group_texts:
            use_truncation = True
        
        tokenize_fn = conversation_tokenize_function
        tokenize_fn_kwargs = {
            "data_args": data_args,
            "tokenizer": self.tokenizer,
            "column_names": column_names,
        }
        if "conversation" in dataset_type:
            tokenize_fn_kwargs["conversation_template"] = conversation_template
        else:
            tokenize_fn_kwargs["label_columns"] = label_columns
            tokenize_fn_kwargs["tokenized_column_order"] = tokenized_column_order
            tokenize_fn_kwargs["add_special_tokens"] = add_special_tokens
            tokenize_fn_kwargs["use_truncation"] = use_truncation
                           
        tokenize_kwargs = {}
        if not data_args.streaming:
            fingerprint = hashlib.md5(
                (
                    raw_datasets.get_fingerprint()
                    + str(self.tokenizer)
                    + f'###padding_side={self.tokenizer.padding_side}'
                    + ('###conversation_template=' + str(conversation_template) if "conversation" in dataset_type else "")
                    + f'###disable_group_texts={data_args.disable_group_texts}'
                    + f'###block_size={data_args.block_size}'
                ).encode("utf-8")
            ).hexdigest()
            tokenize_kwargs = {
                "num_proc": data_args.preprocessing_num_workers,
                "load_from_cache_file": not data_args.overwrite_cache,
                "desc": "Running tokenizer on dataset",
                "new_fingerprint": fingerprint,
            }

        tokenized_datasets = raw_datasets.map(
            tokenize_fn,
            batched=True,
            remove_columns=column_names,
            fn_kwargs=tokenize_fn_kwargs,
            **tokenize_kwargs
        )

        return tokenized_datasets

def conversation_tokenize_function(
    examples, 
    data_args: DatasetArguments,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
    column_names,
    conversation_template: ConversationTemplateForTool,
) -> Dict:
    """Handels conversation datasets tokenization
    """
    num_example = len(examples[column_names[0]])
    token_dict = {
        "input_ids": [[] for _ in range(num_example)],
        "attention_mask": [[] for _ in range(num_example)],
        "labels": [[] for _ in range(num_example)],
    }
    with CaptureLogger(tok_logger) as cl:
        for i in range(len(examples["messages"])):
            messages = examples["messages"][i]
            system = examples.get("system", [None] * num_example)[i]
            tools = examples.get("tools", [None] * num_example)[i]
            if len(messages) < 2 or messages[0]['role'] != CONVERSATION_ROLE_NAMES['user']:
                tok_logger.warning(
                    "Invalid instance encountered. Either the conversation has less than "
                    "one round or the first message is not from the user."
                )
                continue
        
            if len(messages) % 2 != 0:
                logger.warning(
                    "The number of messages is not even, the last message will be ignored."
                )
                messages = messages[:-1]
                
            encoded_conversation = conversation_template.encode_conversation(
                tokenizer=tokenizer,
                messages=messages,
                system=system,
                tools=tools,
            )

            input_ids, labels = [], []
            for turn_idx, conversation_tuple in enumerate(encoded_conversation):
                if len(conversation_tuple) == 2:
                    user_input = conversation_tuple[0]
                    assistant_result = conversation_tuple[1]
                    input_ids += user_input + assistant_result
                    if data_args.train_on_prompt:
                        labels += user_input + assistant_result
                    else:
                        labels += [-100] * len(user_input) + assistant_result
                elif len(conversation_tuple) == 4:
                    user_input = conversation_tuple[0]
                    function_result = conversation_tuple[1]
                    observation_input = conversation_tuple[2]
                    assistant_result = conversation_tuple[3]
                    input_ids += user_input + function_result + observation_input + assistant_result
                    if data_args.train_on_prompt:
                        labels += user_input + function_result + observation_input + assistant_result
                    else:
                        labels += [-100] * len(user_input) + function_result + [-100] * len(observation_input) + assistant_result
                else:
                    logger.warning("The number of roles in conversation is not appropriate")
                
            token_dict["input_ids"][i].extend(input_ids)
            token_dict["attention_mask"][i].extend([1] * len(input_ids))
            token_dict["labels"][i].extend(labels)

    if data_args.disable_group_texts:
        token_dict = blocking(
            token_dict=token_dict,
            block_size=data_args.block_size,
            model_max_length=tokenizer.model_max_length,
            pad_token_id=tokenizer.pad_token_id,
            padding_side=tokenizer.padding_side,
        )

    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    return token_dict


def train():

    # Initialize args
    ## Prepare training_args
    pipeline_name = "finetuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)
    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses() 
    print("Model args", model_args)
    print("data_args", data_args)
    print("training_args", pipeline_args)

    # Init model
    model = HFDecoderModelForTool(model_args)

    # Process data
    dataset = Dataset(data_args)

    # Finetune
    finetuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    tuned_model = finetuner.tune(model=model, dataset=dataset)

if __name__ == "__main__":
    train()