from dataclasses import dataclass, field
from typing import Optional, List
import os

from lmflow.args import DatasetArguments

@dataclass
class T2IDatasetArguments(DatasetArguments):
    """Arguments for T2I dataset"""
    
    image_folder: Optional[str] = field(
        default=None, metadata={"help": "The folder of the image file."}
    )
    
    image_size: Optional[int] = field(
        default=512, metadata={"help": "The size of the image."}
    )
    
    image_crop_type: Optional[str] = field(
        default="center", metadata={"help": "The type of image crop."}
    )
    
    text_embedding_type: Optional[str] = field(
        default="raw", metadata={"help": "How to get text embedding."}
    )
    
    is_t2i: Optional[bool] = field(
        default=True, metadata={"help": "Flag for the modality type."}
    )
    
    def __post_init__(self):
        def check_extension(file_path: str, extension: str):
            assert file_path.split(".")[-1] == extension, f"The file must be a {extension} file."
        
        
        if self.dataset_path is None or self.image_folder is None:
            raise ValueError("The dataset_path, image_folder must be provided.")
            
        else:
            if self.train_file is None:
                if os.path.exists(os.path.join(self.dataset_path, "train.json")):
                    self.train_file = "train.json"
                else:
                    raise ValueError("The train_file must be provided.")
        
            check_extension(self.train_file, "json")
            if (self.validation_file is not None and self.test_file is None)\
                or (self.validation_file is None and self.test_file is not None):
                    same_file = self.validation_file if self.validation_file is not None else self.test_file
                    self.validation_file = same_file
                    self.test_file = same_file
            if self.validation_file is not None:
                check_extension(self.validation_file, "json")
                if not os.path.exists(os.path.join(self.dataset_path, self.validation_file)):
                    self.validation_file = None
            if self.test_file is not None:
                check_extension(self.test_file, "json")
                if not os.path.exists(os.path.join(self.dataset_path, self.test_file)):
                    self.test_file = None

@dataclass           
class DiffuserModelArguments:
    """Arguments for T2I model"""
    
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model name or path."}
    )
    
    model_type: Optional[str] = field(
        default=None, metadata={"help": "The model type."}
    )
    
    # torch_dtype: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
    #             "dtype will be automatically derived from the model's weights."
    #         ),
    #         "choices": ["auto", "bfloat16", "float16", "float32"],
    #     },
    # )
    
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to lora."},
    )
    
    lora_r: int = field(
        default=8,
        metadata={"help": "the rank of the lora parameters. The smaller lora_r is , the fewer parameters lora has."},
    )
    lora_alpha: int = field(
        default=8,
        metadata={
            "help": "Merging ratio between the fine-tuned model and the original. This is controlled by a parameter called alpha in the paper."},
    )
    lora_target_modules: List[str] = field(
        default=None, metadata={"help": "Modules to apply lora."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate in lora.linear."},
    )
  
@dataclass  
class DiffuserTunerArguments:
    """Arguments for T2I finetuner"""
    
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "The output directory."}
    )
    
    logging_dir: Optional[str] = field(
        default="logs", metadata={"help": "The logging directory."}
    )
    
    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "Overwrite the content of the output directory."}
    )
    
    mixed_precision: str = field(
        default="no", metadata={"help": "Whether to use mixed precision."}
    )
    
    do_train: bool = field(
        default=True, metadata={"help": "Whether to run training."}
    )
    
    num_train_epochs: Optional[int] = field(
        default=50, metadata={"help": "The number of training epochs."}
    )
    
    train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "The number of batch size in training."}
    )
    
    learning_rate: Optional[float] = field(
        default=1e-4, metadata={"help": "The learning rate."}
    )
    
    weight_decay: Optional[float] = field(
        default=0.0, metadata={"help": "The weight decay."}
    )
    
    do_valid: bool = field(
        default=True, metadata={"help": "Whether to run evaluation."}
    )
    
    do_test: bool = field(
        default=True, metadata={"help": "Whether to run testing."}
    )
    
    valid_steps: Optional[int] = field(
        default=50, metadata={"help": "The evaluation steps."}
    )
    
    valid_seed: Optional[int] = field(
        default=42, metadata={"help": "The seed for validation."}
    )
    
    test_seed: Optional[int] = field(
        default=42, metadata={"help": "The seed for testing."}
    )
    
    save_steps: Optional[int] = field(
        default=500, metadata={"help": "The saving steps."}
    )
    
    save_total_limit: Optional[int] = field(
        default=None, metadata={"help": "The total number of checkpoints to save."}
    )