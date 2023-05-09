# LMFlow Benchmark: An Automatic Evaluation Framework for Open-Source LLMs

May 9, 2023


## Introduction

Evaluation of a chat-style Large Language Model (LLM) has been a huge challenge since the breakthrough of ChatGPT. On the one hand, researchers and engineers need a reliable way to compare two models and decide which model to choose under a certain application scenario. On the other hand, they have to monitor the model performance during the training of an LLM to avoid performance issues such as forgetting.

Recent work of Vicuna introduces comparison methods of human evaluation, a.k.a. Chatbot Arena. They also pioneered the evaluation method by invoking GPT-4 to compare the outputs of two models. However, those methods require expensive human labeling or GPT-4 API calls, which are neither scalable nor convenient for LLM development.

In this article, we introduce the LMFlow benchmark, which provides a cheap and easy-to-use evaluation framework that can help reflect different aspects of LLMs. We have also open-sourced the dataset and the code, so that everyone in the LLM community can use those toolkits to evaluate, monitor and compare different LLMs.



## Metric

In our evaluation framework, Negative Log Likelihood (NLL) is used for evaluating LLM

![eq](../_static/eq.png)


which corresponds to the LLM model’s prediction probability over a corpus set given their contexts. If the corpus set itself indicates a certain type of LLM ability, such as multi-round conversation, instruction following, math problem solving, role-playing, then NLL on those corpora can provide quantitative metrics to reflect those abilities.


The key idea behind NLL, is that **Generation ability is positively correlated with prediction ability.**

For instance, an LLM who performs well in essay writing should have no problem understanding and predicting a reference human essay, just like human chess masters performing well at memorizing an endgame on a chessboard.

Besides NLL, another similar and commonly used metric in NLP is Perplexity (PPL):



## Chat Performance 


## CommonSense Performance



## Instruction Following


## Conclusion

In this article, we introduce LMFlow’s evaluation framework, which uses NLL metric to reflect LLM models’ ability. NLL provides a good metric to evaluate different aspects of a LLM model. According to our evaluation results, Robin-7b achieves on-par performance when compared with Vicuna-13b. As our Robin-7b model is finetuned with different sources of dataset instead of sharegpt only, this shows that Vicuna can be further improved or surpassed with smaller-sized models and better dataset.

The checkpoint of Robin-7b is now available for engineers and researchers to download and use (TODO link @Jipeng). Its effectiveness demonstrates that a multi-aspect evaluation is indeed essential to the development of LLMs. Hope the LMFlow benchmark can be a cornerstone to building more strong, diverse, and robust models.


## Reference

Vicuna Chatbot Arena: https://chat.lmsys.org/?arena 

lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness 
