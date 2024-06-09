from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

# retrieval usage
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from pathlib import Path
import re
import os
import argparse
import logging
logging.getLogger().setLevel(logging.ERROR)  # hide warning log


class LangchainChatbot:
    def __init__(self,
                 model_name_or_path: str,
                 provider: str):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are a helpful chatbot."),
                MessagesPlaceholder(variable_name="history"),
                MessagesPlaceholder(variable_name="retriever", optional=True),
                HumanMessagePromptTemplate.from_template("{input}")
            ]
        )
        self.model_name_or_path = model_name_or_path
        self.provider = provider
        self.check_valid_provider()
        self.model = self.get_model()
        self.retriever_url = None
        self.retriever_file = None
        self.memory = {}
        self.runnable: Runnable = self.prompt | self.model
        self.llm_chain = RunnableWithMessageHistory(
            self.runnable,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def check_valid_provider(self):
        provider = self.provider
        model_name_or_path = self.model_name_or_path
        if provider == "openai" and 'gpt' in model_name_or_path:
            if os.getenv("OPENAI_API_KEY") is None:
                raise OSError("OPENAI_API_KEY environment variable is not set.")
        elif provider == "anthropic" and 'claude' in model_name_or_path:
            if os.getenv("ANTHROPIC_API_KEY") is None:
                raise OSError("ANTHROPIC_API_KEY environment variable is not set.")
        elif provider == "google" and 'gemini' in model_name_or_path:
            if os.getenv("GOOGLE_API_KEY") is None:
                raise OSError("GOOGLE_API_KEY environment variable is not set.")
        elif provider == "huggingface":
            if os.getenv("HUGGINGFACEHUB_API_TOKEN") is None:
                raise OSError("HUGGINGFACEHUB_API_TOKEN environment variable is not set.")
        else:
            raise ValueError("Invalid provider or model_name_or_path.")

    def set_retriever_url(self, url, chunk_size, chunk_overlap):
        loader = WebBaseLoader(url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        self.retriever_url = vectorstore.as_retriever(k=4)

    def set_retriever_file(self, file, chunk_size, chunk_overlap):
        loader = TextLoader(file, encoding='utf-8')
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        self.retriever_file = vectorstore.as_retriever(k=4)

    def get_model(self):
        provider = self.provider
        model_name_or_path = self.model_name_or_path
        if provider == "openai":
            model = ChatOpenAI(model=model_name_or_path)
        elif provider == "anthropic":
            model = ChatAnthropic(model=model_name_or_path)
        elif provider == "google":
            model = ChatGoogleGenerativeAI(model=model_name_or_path)
        elif provider == "huggingface":
            model = HuggingFacePipeline.from_model_id(model_id=model_name_or_path, task="text-generation")
            # model = HuggingFaceEndpoint(repo_id=model_name_or_path)
        else:
            raise ValueError("Invalid provider.")
        return model

    def chat_with_chatbot(self, human_input, session_id):
        retriever_search = []
        if self.retriever_url:
            retriever_search.extend(self.retrieve_by_url(human_input))
        if self.retriever_file:
            retriever_search.extend(self.retrieve_by_file(human_input))

        response = self.llm_chain.invoke({"input": human_input,
                                          "retriever": retriever_search},
                                         config={"configurable": {"session_id": session_id}})
        return response if self.provider == "huggingface" else response.content

    def retrieve_by_url(self, query):
        return [re.sub('\n+', '\n', dict(result)['page_content']) for result in self.retriever_url.invoke(query)]

    def retrieve_by_file(self, query):
        return [re.sub('\n+', '\n', dict(result)['page_content']) for result in self.retriever_file.invoke(query)]

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.memory:
            self.memory[session_id] = ChatMessageHistory()
        return self.memory[session_id]


def get_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--model-name-or-path", type=str, help="Model name or path"
    )
    parser.add_argument(
        "--provider", type=str, help="Provider of the model"
    )
    parser.add_argument(
        "--set-url", action="store_true", help="Set a URL for retrieval if enabled"
    )
    parser.add_argument(
        "--set-txt", action="store_true", help="Set a single text file for retrieval if enabled"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=400, help="Chunk size for splitting documents."
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=20, help="Chunk overlap for splitting documents."
    )
    parser.add_argument(
        "--session-id", type=str, default="demo", help="Session id of this chat"
    )
    parser.add_argument(
        "--save-history", action="store_true", help="Save chat history if enabled"
    )
    parser.add_argument(
        "--save-dir", type=Path, default=Path("tmp", "chat_history"), help="Directory to store chat history"
    )
    return parser


def main(model_name_or_path: str,
         provider: str,
         set_url: bool,
         set_txt: bool,
         chunk_size: int,
         chunk_overlap: int,
         session_id: str,
         save_history: bool,
         save_dir: Path
         ):
    chatbot = LangchainChatbot(model_name_or_path=model_name_or_path,
                               provider=provider)
    if set_url:
        url = input("Please enter the url: ")
        chatbot.set_retriever_url(url, chunk_size, chunk_overlap)
    if set_txt:
        file = input("Please enter the text file path: ")
        chatbot.set_retriever_file(file, chunk_size, chunk_overlap)
    while True:
        human_input = input("User: ")
        if human_input == "exit":
            break
        response = chatbot.chat_with_chatbot(human_input, session_id)
        print(f"Chatbot: {response}")
    if save_history:
        if '/' in model_name_or_path:
            model_name_or_path = Path(model_name_or_path).parts[-1]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = Path(save_dir, f"{model_name_or_path}_{session_id}.txt")
        with open(save_path, 'w') as file:
            file.write(str(chatbot.memory[session_id].messages))


if __name__ == "__main__":
    args = get_cli().parse_args()
    main(**vars(args))
