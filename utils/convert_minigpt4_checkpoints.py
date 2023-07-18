import argparse
import os.path as osp
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Convert checkpoint from MiniGPT4")
    parser.add_argument("--model_path", type=str, help="the model path for the to convert checkpoint")
    parser.add_argument("--save_path", default=None, type=str, help="the save path for converted checkpoint")
    args = parser.parse_args()
    return args





if __name__ == "__main__":
    args = parse_args()
    model = torch.load(args.model_path)
    model = model['model']
    new_model = {}
    for key, item in model.items():
        key = key.replace("Qformer", "qformer")
        key = key.replace("llama_proj", "language_projection")
        key = key.replace("llama_model.model", "language_model.model")
        new_model[key] = item
    if args.save_path is None:
        end_string = osp.splitext(args.model_path)
        save_path = osp.dirname(args.model_path) + "/" + \
                    osp.basename(args.model_path).replace(".pth", "") + \
                    "-converted" + osp.splitext(args.model_path)[-1]
    else:
        save_path = args.save_path
    print("save_path: {}".format(save_path))

    torch.save(new_model, save_path)
