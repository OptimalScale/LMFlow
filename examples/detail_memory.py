import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_config,
    get_peft_model,
    prepare_model_for_kbit_training
)
import time
import sys

LISA = True if sys.argv[3] == "1" else False
LORA = True if sys.argv[4] == "1" else False
lora_rank = int(sys.argv[5])
# Check if the model name is provided as a command-line argument
if len(sys.argv) < 6:
    print("Usage: python script_name.py <model_name>")
    sys.exit(1)


print("*"*50)
print("Script started")
print("model            : ", sys.argv[1])
print("token_length     : ", sys.argv[2])
print("LISA             : ", LISA)
print("LORA             : ", LORA)
print("lora_rank        : ", lora_rank)
# Model initialization
model_name = sys.argv[1]
token_length = sys.argv[2]
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Measure memory usage of the weights
model.to('cuda')  # Ensure the model is on GPU
if LISA:
    # Only activate two layers
    for param in model.model.layers.parameters():
        param.requires_grad = False
    for param in model.model.layers[-1].parameters():
        param.requires_grad = True
    for param in model.model.layers[-2].parameters():
        param.requires_grad = True

if LORA:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[ "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics
weight_memory = torch.cuda.memory_allocated()

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Execute a forward pass and measure the time
sentence_2048_tokens = 'The OpenAI API is powered by GPT-3 language models which can be coaxed to perform natural language tasks using carefully engineered text prompts. But these models can also generate outputs that are untruthful, toxic, or reflect harmful sentiments. This is in part because GPT-3 is trained to predict the next word on a large dataset of Internet text, rather than to safely perform the language task that the user wants. In other words, these models aren’t aligned with their users.\nTo make our models safer, more helpful, and more aligned, we use an existing technique called reinforcement learning from human feedback (RLHF). On prompts submitted by our customers to the API,A[A]We only use prompts submitted through the Playground to an earlier version of the InstructGPT models that was deployed in January 2021. Our human annotators remove personal identifiable information from all prompts before adding it to the training set.our labelers provide demonstrations of the desired model behavior, and rank several outputs from our models. We then use this data to fine-tune GPT-3.The resulting InstructGPT models are much better at following instructions than GPT-3. They also make up facts less often, and show small decreases in toxic output generation. Our labelers prefer outputs from our 1.3B InstructGPT model over outputs from a 175B GPT-3 model, despite having more than 100x fewer parameters. At the same time, we show that we don’t have to compromise on GPT-3’s capabilities, as measured by our model’s performance on academic NLP evaluations.These InstructGPT models, which have been in beta on the API for more than a year, are now the default language models accessible on our API.B[B]The InstructGPT models deployed in the API are updated versions trained using the same human feedback data. They use a similar but slightly different training method that we will describe in a forthcoming publication.We believe that fine-tuning language models with humans in the loop is a powerful tool for improving their safety and reliability, and we will continue to push in this direction.This is the first time our alignment research, which we’ve been pursuing for several years,1,2,3 has been applied to product.Yesterday we announced our next-generation Gemini model: Gemini 1.5. In addition to big improvements to speed and efficiency, one of Gemini 1.5’s innovations is its long context window, which measures how many tokens — the smallest building blocks, like part of a word, image or video — that the model can process at once. To help understand the significance of this milestone, we asked the Google DeepMind project team to explain what long context windows are, and how this breakthrough experimental feature can help developers in many ways. Context windows are important because they help AI models recall information during a session. Have you ever forgotten someone’s name in the middle of a conversation a few minutes after they’ve said it, or sprinted across a room to grab a notebook to jot down a phone number you were just given? Remembering things in the flow of a conversation can be tricky for AI models, too — you might have had an experience where a chatbot “forgot” information after a few turns. That’s where long context windows can help. Previously, Gemini could process up to 32,000 tokens at once, but 1.5 Pro — the first 1.5 model we’re releasing for early testing — has a context window of up to 1 million tokens — the longest context window of any large-scale foundation model to date. In fact, we’ve even successfully tested up to 10 million tokens in our research. And the longer the context window, the more text, images, audio, code or video a model can take in and process. "Our original plan was to achieve 128,000 tokens in context, and I thought setting an ambitious bar would be good, so I suggested 1 million tokens," says Google DeepMind Research Scientist Nikolay Savinov, one of the research leads on the long context project. “And now we’ve even surpassed that in our research by 10x.” To make this kind of leap forward, the team had to make a series of deep learning innovations. Early explorations by Pranav Shyam offered valuable insights that helped steer our subsequent research in the right direction. “There was one breakthrough that led to another and another, and each one of them opened up new possibilities,” explains Google DeepMind Engineer Denis Teplyashin. “And then, when they all stacked together, we were quite surprised to discover what they could do, jumping from 128,000 tokens to 512,000 tokens to 1 million tokens, and just recently, 10 million tokens in our internal research.” The raw data that 1.5 Pro can handle opens up whole new ways to interact with the model. Instead of summarizing a document dozens of pages long, for example, it can summarize documents thousands of pages long. Where the old model could help analyze thousands of lines of code, thanks to its breakthrough long context window, 1.5 Pro can analyze tens of thousands of lines of code at once. “In one test, we dropped in an entire code base and it wrote documentation for it, which was really cool,” says Google DeepMind Research Scientist Machel Reid. “And there was another test where it was able to accurately answer questions about the 1924 film Sherlock Jr. after we gave the model the entire 45-minute movie to ‘watch.’” 1.5 Pro can also reason across data provided in a prompt. “One of my favorite examples from the past few days is this rare language — Kalamang — that fewer than 200 people worldwide speak, and there one grammar manual about it,” says Machel. The model can speak it on its own if you just ask it to translate into this language, but with the expanded long context window, you can put the entire grammar manual and some examples of sentences into context, and the model was able to learn to translate from English to Kalamang at a similar level to a person learning from the same content.”Gemini 1.5 Pro comes standard with a 128K-token context window, but a limited group of developers and enterprise customers can try it with a context window of up to 1 million tokens via AI Studio and Vertex AI in private preview. The full 1 million token context window is computationally intensive and still requires further optimizations to improve latency, which we’re actively working on as we scale it out. And as the team looks to the future, they’re continuing to work to make the model faster and more efficient, with safety at the core. They’re also looking to further expand the long context window, improve the underlying architectures, and integrate new hardware improvements. “10 million tokens at once is already close to the thermal limit of our Tensor Processing Units — we dont know where the limit is yet, and the model might be capable of even more as the hardware continues to improve,” says Nikolay.The team is excited to see what kinds of experiences developers and the broader community are able to achieve, too. “When I first saw we had a million tokens in context, my first question was, ‘What do you even use this for?’” says Machel. “But now, I think people’s imaginations are expanding, and they’ll find more and more creative ways to use these new capabilities.Long-context question answering (QA) tasks require reasoning over a long document or multiple documents. Addressing these tasks often benefits from identifying a set of evidence spans (e.g., sentences), which provide supporting evidence for answering the question. In this work, we propose a novel method for equipping long-context QA models with an additional sequence-level objective for better identification of the supporting evidence. We achieve this via an additional contrastive supervision signal in finetuning, where the model is encouraged to explicitly discriminate supporting evidence sentences from negative ones by maximizing question-evidence similarity. The proposed additional loss exhibits consistent improvements on three different strong long-context transformer models, across two challenging question answering benchmarks – HotpotQA and QAsper.Actively and judiciously select the most helpful questions for LLMs“Existing CoT studies largely rely on a fixed set of human-annotated exemplars, which are not necessarily the most effective ones. A good performance requires human prompt engineering which is costly. We identify the human prompt engineering as two complementary components: question selection and prompt template engineering. In this paper, we offer a solution to the key problem of determining which questions are the most important and helpful ones to annotate from a pool of task-specific queries. By borrowing ideas from the related problem of uncertainty-based active learning, we introduce several metrics to characterize the uncertainty so as to select the most uncertain questions for annotation. Experimental results demonstrate the superiority of our proposed method on eight complex reasoning tasks. With text-davinci-002, active-prompt improves upon  by 7 (67.9->74.9).'
inputs = tokenizer(sentence_2048_tokens, return_tensors='pt', max_length=int(token_length),truncation=True).to('cuda')
labels = inputs['input_ids']
# store the gpu memory for inputs
input_memory = torch.cuda.memory_allocated() - weight_memory

torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats to measure forward pass accurately
start = time.time()
outputs = model(**inputs, labels=labels)
activation_memory = torch.cuda.memory_allocated() - weight_memory 
forward_time = time.time() - start

# Execute a backward pass and measure the time
start = time.time()
loss = outputs.loss
# torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats again before backward pass
loss.backward()
gradient_memory = torch.cuda.memory_allocated() - weight_memory  # activation memory will free automatically

optimizer.step()
optimizer_memory = torch.cuda.memory_allocated() - gradient_memory - weight_memory 
backward_time = time.time() - start


# Total memory
total_memory = torch.cuda.memory_allocated()

print(f"Weight memory    : {weight_memory / 1e6} MB")
print(f"Input memory     : {input_memory / 1e6} MB")
print(f"Activation memory: {activation_memory / 1e6} MB")
print(f"Gradient memory  : {gradient_memory / 1e6} MB")
print(f"Optimizer memory : {optimizer_memory / 1e6} MB")
print(f"Total memory     : {total_memory / 1e6} MB\n")
print(f"Forward time     : {forward_time} s")
print(f"Backward time    : {backward_time} s")
print("*"*50)
