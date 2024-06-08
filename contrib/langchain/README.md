## Langchain

### Setup

```
pip install langchain
pip install langchain-openai langchain-anthropic langchain-google-genai langchain-huggingface langchain-chroma langchain-community
```
     
### Run Chatbot

To run the script, go to the root of this repo and use the following command:

```
python contrib/langchain/retrieval_chatbot.py [options]
```

### Command-Line Arguments
- `--model-name-or-path` - Specifies the name or path of the model used for generating responses.
- `--provider` - Supports the following providers: `openai`, `anthropic`, `google`, and `huggingface`.
- `--set-url` - The chatbot to retrieve content from a specified URL if enabled.
- `--save-history` - Saves the chat history in the `chat_history` directory if enabled.

### Example Usage

- Inference with `mistralai/Mistral-7B-Instruct-v0.2` and specified url
```commandline
python contrib/langchain/retrieval_chatbot.py --provider "huggingface" --model-name-or-path "mistralai/Mistral-7B-Instruct-v0.2" --set-url
```
