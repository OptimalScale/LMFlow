# Finetune

```python
import sys

from transformers import HfArgumentParser

from lmflow.args import (
    ModelArguments,
    DatasetArguments,
    AutoArguments,
)

from lmflow.datasets.dataset import Dataset
from lmflow.models.tunable_models import TunableModel
from lmflow.pipeline.auto_pipeline import AutoPipeline


def main():
    # Parses arguments
    pipeline_name = "finetuner"
    PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

    parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

    # TODO: deepspeed config initialization

    # Initialization
    finetuner = AutoPipeline.get_pipeline(
        pipeline_name=pipeline_name,
        model_args=model_args,
        data_args=data_args,
        pipeline_args=pipeline_args,
    )
    dataset = Dataset(data_args)
    model = TunableModel(model_args)

    # Tokenization and text grouping must be done in the main process
    with pipeline_args.main_process_first(desc="dataset map tokenization"):
        tokenized_dataset = model.tokenize(dataset)
        lm_dataset = finetuner.group_text(
            tokenized_dataset,
            model_max_length=model.get_max_length(),
        )

    # Finetuning
    tuned_model = finetuner.tune(model=model, lm_dataset=lm_dataset)

```
