# Multi-node Training with LMFlow

### Environment setup

If NFS is available, we only have to setup this once in the shared folder. Otherwise, the following procedure is required for every node in the cluster.

We demonstrate the setup of multi-node environment in Ubuntu as an example.

#### Step 1: Set up single-node configurations
Following the main `README` of LMFlow.

#### Step 2: Prepare multi-node communication tools
```
# Require root permission
apt-get install pdsh
```

#### Step 3: Set up `~/.bashrc`

```
# In ~/.bashrc
export NCCL_SOCKET_IFNAME=eth0
export PDSH_RCMD_TYPE=ssh
```

LMFlow majorly utilizes the stack of `deepspeed -> PDSH + torch.distributed (which requires NCCL)` for multi-node communication. Here we specify the default communication protocol of `pdsh` with `ssh`.

#### Step 4: Establish SSH trust between servers

It is important that `ssh` across different nodes should require no password. To establish trust from server A to server B, you may follow the steps below

* **4.1. In server A:** Run  `cat ~/.ssh/id_rsa.pub`. If `~/.ssh/id_rsa.pub` does not exist, you can always run `ssh-keygen` to generate the key pair.
* **4.2 In server B:** After obtaining the content, append the public key into `~/.ssh/authorized_keys
`. If the file does not exist, you may create the file.
* **4.3 In Server A:** Run `ssh {server_B_ip}` to check the ssh trust, also to make server B included in the known hosts of server A during this step.

If NFS is available, you will only need to do this once, i.e. copy the content of `~/.ssh/id_rsa.pub` into `~/.ssh/authorized_keys` in the shared folder.

However, if NFS is not available, and you have _N_ nodes, then you have to repeat this procedure for _N x N_ times for all node pairs, i.e. 1->1, 1->2, 1->3, ..., 1->N, 2->1, ... (Yes, ssh trust is required even for 1->1).

**Note**: Assume server B's IP address is _ip\_B_, e.g. 100.100.100.100, you may check the ssh trust via `ssh {ip_B}` in server A, e.g. `ssh 100.100.100.100`.

You may also check if `pdsh` is working via `pdsh -w {ip_B} ls`, which should list folders in server B, e.g. `pdsh -w 100.100.100.100`.

### Step 5: Update `hostfile`

Deepspeed requires a hostfile to present information for all nodes (https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node). You can configure the `configs/hostfile` as follows. For example, assume you have 3 nodes with

* Server A, B: 8 GPUs
* Server C: 4 GPUs

where the intranet ip is

* Server A: 100.100.100.1
* Server B: 100.100.100.2
* Server C: 100.100.105.1

then the `configs/hostfile` is like this
```
100.100.100.1 slots=8
100.100.100.2 slots=8
100.100.105.1 slots=4
```

#### Step 6: Check firewalls

Sometimes firewalls between nodes can prevent servers from communicating with each other. The training will just stuck at the initialization stage for more than 20 mins and no one knows what is going on. So make sure check firewall is turned off inside the intranet.

[NCCL test toolkit](https://github.com/NVIDIA/nccl-tests) is highly recommended for debugging purpose, since it is difficulty to debug given just the stuck behavior.

#### Step 7: Running single-node training in each server

Make sure single-node training is fine in each server. Any failure in any server could result in multi-node training failure, and it is hard to debug.

#### Step 8: Run the training script

Now you may run the training script in your main node. The main node will start `torch.distributed` in each client node and launch the distributed training. The command is similar as follows, assuming we are using the same example as in **Step 5**:
```
./scripts/run_continual_pretrain.sh  \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --data/alpaca/train_conversation \
    --output_model finetuned_llama \
    --deepspeed_args "--master_port=11000 --master_addr=100.100.100.1 --include=100.100.100.1:0,1,2,3,4,5,6,7@100.100.100.2:0,1,2,3,4,5,6,7@100.100.105.1:0,1,2,3 --hostfile=configs/hostfile" \
```
In clusters with limited intranet bandwidth, the initialization can take ~20mins, which is hard to tell from a bad run when the program is actually stucked. So before the actual run, it is always good to use [NCCL test toolkit](https://github.com/NVIDIA/nccl-tests) to ensure that the command is working.
