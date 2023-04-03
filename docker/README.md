# Docker

LMFlow is available as a docker image in Docker Hub, built from the Dockerfile
in this directory, with cuda:11.3.0-cudnn8 (source docker:
nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04).  You need to have at least a
Nvidia 3090 GPU on your machine with cuda driver compatible with cuda:11.3.0 to
run this docker image.

## Install docker with nvidia support

First you may need to install docker with nvidia support. This step requires
root permission. If you don't have one, you may need to contact the system
adminstrator to do that for you.

We provide an example in Ubuntu 20.04. For other operating systems, you may
refer to Nvidia's [Install
Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

```sh
curl https://get.docker.com | sh && sudo systemctl --now enable docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
  && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Pull docker image and run

Use the following command to pull our docker image.

```sh
docker pull optimalscale/lmflow
```

The working directory in docker is `/LMFlow`, where LMFlow (commit:
[fa0e66f94](https://github.com/OptimalScale/LMFlow/tree/fa0e66f94eb5b7bfd624afdf9826b054641e3373))
is cloned and installed.  Use the following command to enter the docker
container, where `./LMFlow/log/finetune` in the container will be mapped to
`./output_dir/log/finetune` on the host machine. You may add more directory
mappings in a similar manner.

```sh
docker run \
  -v ./output_dir/log/finetune:/LMFlow/log/finetune \
  --gpus=all \
  --shm-size=64g \
  -e WANDB_DISABLED=true \
  -it \
  --rm \
  optimalscale/lmflow \
  bash
```

Then you will be able to work inside the docker, just like in a physical
machine. Notice that to use multiple gpus, you need to allocate enough
shared memory. We have setup the dependency for you, so you can directly
run our scripts, e.g.

```
./scripts/run_chatbot.sh
./scripts/run_evaluation.sh

# May need a GPU with --bf16 support, or you can remove --bf16
# and use --fp16 instead
./scripts/run_finetune.sh	
```
