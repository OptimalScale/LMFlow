# LMFlow Benchmark Guide

We support two ways to add evaluation settings in our repo, `NLL Task Setting` and `LM-Evaluation Task Setting`. Below are the details of them: 

# 1. NLL Task Setting
Users can easily create new tasks and evaluate their datasets on 
the provide `nll (Negative Log Likelihood)` metric. 

## Setup

Fork the main repo, clone it, and create a new branch with the name of 
your task, and install the following:

```bash
# After forking...
git clone https://github.com/<YOUR-USERNAME>/LMFlow.git
cd LMFlow
git checkout -b <TASK-NAME>
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
pip install -e .
```
## Create Your Task Dataset File
We provide several available datasets under `data` after running
```sh
cd data && ./download.sh && cd -
```

You can refer to some given evaluation dataset files and create your own.
Also, you may refer to our guide on 
[DATASET](https://optimalscale.github.io/LMFlow/examples/DATASETS.html).

In this step, you will need to decide your answer type like `text2text` 
or `text_only` (Notice that the current `nll` implementation only supports these 
two answer types). We will note the chosen answer type as `<ANSWER_TYPE>`.

After preparing your own `DATASET` file, you can put it under `data` dir
and make a `TASK` dir.

```bash
mkdir <TASK>
mv <DATASET> <TASK>
```

## Task Registration

Note the path of your dataset, `data/<TASK>/<DATASET>`.

Open the file `examples/benchmarking.py`, add your task's info into 
`LOCAL_DATSET_GROUP_MAP`, `LOCAL_DATSET_MAP`, `LOCAL_DATSET_ANSWERTYPE_MAP`

In `LOCAL_DATSET_MAP`, you will need to specify your `DATASET` files' path:

```python
LOCAL_DATSET_MAP ={
    "...":"...",
    "<TASK>":"data/<TASK>/<DATASET>",
}
```

In `LOCAL_DATSET_ANSWERTYPE_MAP`, you will need to specify your task's 
`<ANSWER_TYPE>`:

```python
LOCAL_DATSET_ANSWERTYPE_MAP ={
    "...":"...",
    "<TASK>":"<ANSWER_TYPE>,
}
```

If you only have one task, you can add key-value pair like `"<TASK>":"<TASK>"`
in `LOCAL_DATSET_GROUP_MAP`:
```python
LOCAL_DATSET_GROUP_MAP ={
    "...":"...",
    "<TASK>":"<TASK>",
}
```


If you want to combine several tasks, you may first specify a 
combination name `<TASK_COMBINATION>` and add key-value pair like
`"<TASK_COMBINATION>":"<TASK_1>,<TASK_2>,.."`in `LOCAL_DATSET_GROUP_MAP`.

Remember to separate TASK by `,`:
```python
LOCAL_DATSET_GROUP_MAP ={
    "...":"...",
    "<TASK_COMBINATION>":"<TASK_1>,<TASK_2>,..",
}
```

After finishing changing these items, you can run your own `<TASK>` like:

```bash
deepspeed examples/benchmarking.py \
  --answer_type <ANSWER_TYPE> \
  --use_ram_optimized_load False \
  --model_name_or_path ${model_name} \
  --dataset_name data/<TASK>/<DATASET>\
  --deepspeed examples/ds_config.json \
  --metric nll \
  --prompt_structure "###Human: {input}###Assistant:" \
  | tee ${log_dir}/train.log \
  2> ${log_dir}/train.err 
```

# 2. LM-Evaluation Task Setting

We integrate [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) into 
`benchamrk.py` by directly executing the evaluate commands. Users 
can also use their evaluation by simply changing two items in
`<LM_EVAL_DATASET_MAP>` of `examples/benchmarking.py`. 

Please refer to Eleuther's 
[task-table](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md)
to get exact `<TASK>` name.

Similarly, you can combine several tasks, you may first specify a 
combination name `<TASK_COMBINATION>` and add key-value pair like
`"<TASK_COMBINATION>":"<TASK_1>,<TASK_2>,.."`in `LM_EVAL_DATASET_MAP`.

Also, remember to separate TASK by `,`:

```python
LM_EVAL_DATASET_MAP ={
    "...":"...",
    "<TASK_COMBINATION>":"<TASK_1>,<TASK_2>,..",
}
```

