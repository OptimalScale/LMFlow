from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").half().cuda()
model = model.eval()

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

MAX_INPUT_LEN = 512

MAX_LENGTH = 512

def generate_stream(model, tokenizer, delta_tokens, max_length, **kwargs):

    base_p = len(kwargs["input_ids"][0])
    last_p = base_p
    if len(kwargs["input_ids"][0])>MAX_INPUT_LEN:
        kwargs["input_ids"][0] = kwargs["input_ids"][0][-MAX_INPUT_LEN:]
    kwargs["max_length"] = len(kwargs["input_ids"][0]) + delta_tokens
    while True:

        output_ids = model.generate(kwargs["input_ids"],
                        max_length=kwargs["max_length"],
                        do_sample = kwargs["do_sample"],
                        top_p = kwargs["top_p"],
                        temperature=kwargs["temperature"],
                        repetition_penalty=1.0)
        
        output_seq = output_ids[0].tolist()

        eos_p = len(output_seq)

        return_delta = output_seq[last_p: eos_p]
        return_seq = output_seq[base_p: eos_p]
        yield return_delta, return_seq

        if kwargs["max_length"] >= max_length:
            break

        kwargs["input_ids"] = output_ids
        kwargs["max_length"] = len(output_ids[0]) + delta_tokens
        last_p = eos_p


def chat_stream(model, tokenizer, query: str, history= None, delta_tokens= 2, max_length= 256, do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
    if history is None:
        history = []

    gen_kwargs = {"do_sample": do_sample, "top_p": top_p, "temperature": temperature,**kwargs}
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "### Human:{} ### Assistent:{}".format( old_query, response)
        prompt += "### Human:{} ### Assistent:".format(query)
    input_ids = tokenizer([prompt], return_tensors="pt")
    input_ids = input_ids.to(model.device)
    for delta, seq in generate_stream(model,tokenizer,delta_tokens, max_length, **input_ids, **gen_kwargs):
        if delta:
            delta = tokenizer.decode(delta)
        else:
            delta = ""
        if seq:
            seq = tokenizer.decode(seq)
        else:
            seq = ""
        yield delta, history + [(query, seq)]




def predict(input, max_length, top_p, temperature, history=None):
    if history is None:
        history = []
    for response, history in chat_stream(model,tokenizer, input, history, max_length=max_length*(len(history)+1), top_p=top_p,
                                               temperature=temperature):
        updates = []
        for query, response in history:
            updates.append(gr.update(visible=True, value="" + query))
            updates.append(gr.update(visible=True, value="" + response))
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
        yield [history] + updates

title = """
<h1 align="center">CHAT</h1>
<link rel="stylesheet" href="/path/to/styles/default.min.css">
<script src="/path/to/highlight.min.js"></script>
<script>hljs.highlightAll();</script>
"""
css = """
#user {                                                                         
    float: left;
    position:relative;
    right:5px;
    width:auto;
    min-height:32px;
    max-width: 60%
    line-height: 32px;
    padding: 2px 8px;
    font-size: 14px;
    background:	#77FF00;
    border-radius:5px; /* 圆角 */
    margin:10px 0px;
}
                                             
#chatbot {                                                                      
    float: right;
    position:relative;
    right:5px;
    width:auto;
    min-height:32px;
    max-width: 60%
    line-height: 32px;
    padding: 2px 8px;
    font-size: 14px;
    background:#F8C301;
    border-radius:5px; /* 圆角 */
    margin:10px 0px;
}
"""
with gr.Blocks(css=css) as demo:
    gr.HTML(title)
    state = gr.State([])
    text_boxes = []
    for i in range(MAX_BOXES):
        if i % 2 == 0:
            text_boxes.append(gr.Markdown(visible=False, label="Q:", elem_id="user"))
        else:
            text_boxes.append(gr.Markdown(visible=False, label="A:", elem_id="chatbot"))

    with gr.Column(elem_id = "col_container"):
        with gr.Row():
            with gr.Column(scale=19):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter",
                ).style(container=True)
            with gr.Column(scale=1):
                button = gr.Button("Send")
        with gr.Accordion("Parameters", open=False):
            max_length = gr.Slider(0, MAX_LENGTH, value=128, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.7, step=0.01, label="Temperature", interactive=True)
    button.click(predict, [txt, max_length, top_p, temperature, state], [state] + text_boxes)
demo.queue().launch(share=True)