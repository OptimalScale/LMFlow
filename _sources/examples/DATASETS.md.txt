# Dataset

We provide several available datasets under `data`. You may download them all by running: 
```sh
cd data && ./download.sh all && cd -
```
You can replace `all` with a specific dataset name to only download that dataset (e.g. `./download.sh alpaca`).

Customized datasets are strongly encouraged, since this way users can apply
their own prompt engineering techniques over various source datasets. As long
as the generated dataset following the format below, they can be accepted as
the input of our pipelines :hugs:


## Dataset Format in General

To specify the input for model finetune, users can provide a list of `.json`
files under a specified dataset directory. For example,

```sh
|- path_to_dataset
  |- data_1.json
  |- data_2.json
  |- another_data.json
  |- ...
```

For inference, we currently only support a single `.json` file.

Each json file shall have the following format (three instances with four keys
for example),

```json
{
  "type": "TYPE",
  "instances": [
    {
        "KEY_1": "VALUE_1.1",
        "KEY_2": "VALUE_1.2",
        "KEY_3": "VALUE_1.3",
        "KEY_4": "VALUE_1.4",
    },
    {
        "KEY_1": "VALUE_2.1",
        "KEY_2": "VALUE_2.2",
        "KEY_3": "VALUE_2.3",
        "KEY_4": "VALUE_2.4",
    },
    {
        "KEY_1": "VALUE_3.1",
        "KEY_2": "VALUE_3.2",
        "KEY_3": "VALUE_3.3",
        "KEY_4": "VALUE_3.4",
    },
  ]
}
```

where the `TYPE` indicates the dataset type and defines the set of keys
`{ KEY_1, KEY_2, ... }` and their corresponding interpretations. The list of
supported types are listed as follows.

## Supported Dataset and Detailed Formats

### TextOnly

This is the most common dataset type, which only contains raw texts in each
sample. This type of dataset can be used as the training set for text decoder
models, or the input of decoder models / encoder-decoder models. Its format is
as follows (three instances for example),

```json
{
  "type": "text_only",
  "instances": [
    {  "text": "SAMPLE_TEXT_1" },
    {  "text": "SAMPLE_TEXT_2" },
    {  "text": "SAMPLE_TEXT_3" },
  ]
}
```

For example, `data/example_dataset/train/train_50.json` has the aboved format.

### Text2Text

This is the dataset type mostly used for inferencing, which contains a pair of
texts in each sample. This type of dataset can be used as the training set for
text encoder-decoder models, or question-answer pair for evaluating model
inferences. Its format is as follows (three instances for example),

```json
{
  "type": "text2text",
  "instances": [
    {
        "input": "SAMPLE_INPUT_1",
        "output": "SAMPLE_OUTPUT_1",
    },
    {
        "input": "SAMPLE_INPUT_2",
        "output": "SAMPLE_OUTPUT_2",
    },
    {
        "input": "SAMPLE_INPUT_3",
        "output": "SAMPLE_OUTPUT_3",
    },
  ]
}
```

For example, `data/example_dataset/test/test_13.json` has the aboved format.
