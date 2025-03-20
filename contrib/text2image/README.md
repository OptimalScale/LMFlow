# Fine-tuning Text2Img

Here is a fork function for fine-tuning text2image diffusion model based on diffusers, under the framework of lmflow.

## Environment Preparation

After install the `lmflow`, directly use `pip install -r requirements.txt` for extensive packages of t2i fine-tuning.

## Data Preparation

Here is a tree struct of the required data organization. In detail, under a `dataset_path` *example*, by default, an `img` directory is used for image files, and `train.json`, `valid.json` and `test.json` are used for reference of training, validation and testinig data. The `valid.json` and `test.json` are optional. If one is provided and the other is not, the two files will be set as the same.

```bash
data
└── example
    ├── img
    │   ├── 00.jpg
    │   ├── 01.jpg
    │   ├── 02.jpg
    │   ├── 03.jpg
    │   └── 04.jpg
    ├── train.json
    ├── [valid.json]
    └── [test.json]
```

The `train.json` should be the format as follow:

```json
{
    "type": "text-image",
    "instances": [
        {
            "image": "00.jpg",
            "text": "A photo of a <SKS> dog"
        },
        ...
    ]
}
```

And the `valid.json` and `test.json` should be the format as follow:

```json
{
    "type": "text-only",
    "instances": [
        {
            "text": "A photo of a <SKS> dog in front of Eiffel Tower."
        },
        ...
    ]
}
```

Here is a specific example of the data [dog_t2i_data_example](https://drive.google.com/drive/folders/106ahvIrXbiuZMBw0NuOTjY0vnM_xXARW?usp=sharing)

## Fine-tuning

For convenience, we provide a script `finetune_t2i.sh` for fine-tuning. It can be used as follow:

```bash
bash finetune_t2i.sh \
    model_name_or_path=stabilityai/stable-diffusion-2-1 \
    dataset_path=data/example
```

The `model_name_or_path` is the model name in [huggingface](https://huggingface.co/) or path of the pre-trained model. The `dataset_path` is the path of the dataset, which should be organized as the above tree struct.

There are also some optional arguments for the script:

- `model_type`: The type of the model, which can be `unet` or `transformer`. Default is `unet`. (The `transformer` is not supported yet.)
- `output_dir`: The output directory of the fine-tuned model. Default is `output`.
- `main_port`: The main port of the server. Default is `29500`.
- `img_size`: The size of the image for fine-tuning, validation and testing. Default is `768`.

For more customization, you can refer to the `finetune_t2i.sh` and `finetune_t2i.py`.
