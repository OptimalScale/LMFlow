from flask import Flask, request, make_response, jsonify
from flask import render_template
from flask_cors import CORS
from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
from lmflow.models.hf_decoder_model import HFDecoderModel
import torch.distributed as dist
from transformers import HfArgumentParser
import io
import json
import torch
import os

WINDOW_LENGTH = 512

app = Flask(__name__)
CORS(app)
ds_config_path = "../examples/ds_config.json"
with open (ds_config_path, "r") as f:
    ds_config = json.load(f)

# NOTE: No need to download gpt neo now
# model_name_or_path = '../output_models/gpt_neo2.7B_inst_tuning/'
# NOTE: Directly donwload the checkpoint from hf
model_name_or_path = "OptimalScale/gpt-neo2.7B-inst-tuning"
# lora_path = '../output_models/instruction_ckpt/llama7b-lora/'
model_args = ModelArguments(model_name_or_path=model_name_or_path)

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
model = AutoModel.get_model(model_args, tune_strategy='none', ds_config=ds_config)


@app.route('/predict',methods = ['POST'])
def predict():
    if(request.method == "POST"):
        try:
            user_input = request.get_json()["Input"]
            conversation = request.get_json()["History"]

            history_input = ""
            if(len(conversation) >= 2):
                for i in range(0, len(conversation)-1):
                    history_input = history_input + conversation[i+1]["content"] + "\n"
            if len(model.encode(history_input))> WINDOW_LENGTH:
                inputs = model.encode(history_input)
                inputs = inputs[-WINDOW_LENGTH:]
                history_input = model.decode(inputs)
            
            inputs = model.encode(history_input, return_tensors="pt").to(device=local_rank)
            outputs = model.inference(inputs, max_new_tokens=150,temperature=0.0, do_sample=False)
            text_out = model.decode(outputs[0], skip_special_tokens=True)
            prompt_length = len(model.decode(inputs[0], skip_special_tokens=True,))
            text_out = text_out[prompt_length:].strip("\n")
        except:
            text_out = "There is something wrong, please query again"
    else:
        text_out = "pending"
    return text_out

@app.route('/',methods = ['GET'])
def login():

    return render_template('index.html')


app.run(port = 5000, debug = False)
