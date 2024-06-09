## Langchain

### Setup

```
pip install langchain
pip install langchain-openai langchain-anthropic langchain-google-genai langchain-chroma langchain-community bs4
```
     
### Run Chatbot

To run the script, go to the root of this repo and use the following command:

```
python contrib/langchain/retrieval_chatbot.py [options]
```

### Command-Line Arguments
- `--model-name-or-path` - Specifies the name or path of the model used for generating responses.
- `--provider` - Supports the following providers: `openai`, `anthropic`, `google`, and `huggingface`.
- `--set-url` - Retrieve content from a specified URL if enabled.
- `--set-txt` - Retrieve content from a local txt file if enabled.
- `--session-id` - Session id of this chat, default: `demo`.
- `--save-history` - Saves the chat history if enabled.
- `--save-dir` - Directory to store chat history, default: `tmp/chat_history`

### Example Usage

- Inference with `gpt-4o`, specified url and txt file
```
cd data && ./download.sh example_doc_for_retrieval.txt && cd -
python contrib/langchain/retrieval_chatbot.py --provider "openai" --model-name-or-path "gpt-4o" --set-url --set-txt
```
- Then set the url and txt file as follows:
```
Please enter the url: https://optimalscale.github.io/LMFlow/index.html
Please enter the text file path: data/example_doc_for_retrieval.txt
```