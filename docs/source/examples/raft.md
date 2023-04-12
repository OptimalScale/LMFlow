# Reward rAnked FineTuning (RAFT)

## Introduction

 Extensive unsupervised training data used in generative foundational models can lead to implicit biases. Such biases can lead to low-quality samples, biased results, and unfairness, which can have substantial consequences. Therefore, aligning generative foundational models with human ethics and preferences has become a crucial procedure to ensure their well-behaved use in real-world scenarios. Previous works have primarily relied on Reinforcement Learning from Human Feedback (RLHF) to overcome this issue. RLHF tunes generative models using RL algorithms with a reward model supervised by human feedback. Despite the feasibility of RLHF, the inefficiency and instability of RL algorithms often pose significant challenges in aligning generative models. Therefore, there is an urgent need to streamline and enhance the alignment pipeline. In this work, we propose a generic framework, Reward rAnked FineTuning (RAFT), to align generative models. Given a reward model and a sufficient number of samples, we rank the best samples and reject ill-behaved ones to construct a streaming dataset. This dataset can then be used to align the generative model, and the procedure can be done under both offline and online settings. Furthermore, the sample generation process is gradient-free, so that RAFT  
  supports black-box generators. 

  Our experiments demonstrate that the RAFT algorithm performs well on both large language models.

## Algorithm

<img src="../_static/raft.png" alt="RAFT" style="width: 100%; min-width: 300px; display: block; margin: auto;">


## Demo 



