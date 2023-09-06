import argparse
from huggingface_hub import hf_hub_download
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Download huggingface file")
    parser.add_argument("--repo_id", type=str, help="the repo id")
    parser.add_argument("--filename", default=None, type=str, help="the file name for the download file")
    parser.add_argument("--target_path", default="./", type=str, help="the target path for the download file")
    parser.add_argument("--repo_type", default="dataset", type=str, help="the repo type")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    print("Start downloading repo {} filename: {}".format(
        args.repo_id, args.filename))
    args.target_path = os.path.abspath(args.target_path)
    source_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename, repo_type=args.repo_type)
    os.makedirs(args.target_path, exist_ok=True)
    target_path = os.path.join(args.target_path, args.filename)
    shutil.copyfile(source_path, target_path)
    print("Finish downloading repo {} filename: {}".format(
        args.repo_id, args.filename))
