from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

# retriever usage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import re
import os
import argparse
from getpass import getpass


class LangchainChatbot:
    def __init__(self,
                 model_name_or_path='gpt-3.5-turbo'):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are a helpful chatbot."),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="retriever", optional=True)
            ]
        )
        self.model_name_or_path = model_name_or_path
        self.model = self.get_model()
        self.retriever = None
        self.memory = {}
        self.runnable = self.prompt | self.model
        self.llm_chain = RunnableWithMessageHistory(
            self.runnable,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def set_retriever_url(self, url):
        loader = WebBaseLoader(url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        self.retriever = vectorstore.as_retriever(k=4)

    def get_model(self):
        model_name_or_path = self.model_name_or_path
        if 'gpt' in model_name_or_path:
            model = ChatOpenAI(model=model_name_or_path)
        elif 'claude' in model_name_or_path:
            model = ChatAnthropic(model=model_name_or_path)
        elif 'gemini' in model_name_or_path:
            model = ChatGoogleGenerativeAI(model=model_name_or_path)
        else:
            llm = HuggingFaceHub(repo_id=model_name_or_path)
            model = ChatHuggingFace(llm=llm)
        return model

    def chat_with_chatbot(self, human_input):
        if self.retriever:
            retriever_search = self.retrieve_by_retriever(human_input)
            response = self.llm_chain.invoke({"input": human_input,
                                              "retriever": [retriever_search]},
                                             config={"configurable": {"session_id": "abc123"}}).content
        else:
            response = self.llm_chain.invoke({"input": human_input},
                                             config={"configurable": {"session_id": "abc123"}}).content
        return response

    def retrieve_by_retriever(self, query):
        return '\n'.join(re.sub('\n+', '\n', dict(result)['page_content']) for result in self.retriever.invoke(query))

    def retrieve_by_memory(self, keyword):
        return [msg.content for msg in self.memory.chat_memory.messages if keyword in msg.content]

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.memory:
            self.memory[session_id] = ChatMessageHistory()
        return self.memory[session_id]


def get_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--model-name-or-path", type=str, help="Model name"
    )
    parser.add_argument(
        "--set-url", action="store_true", help="URL for retrieval"
    )
    parser.add_argument(
        "--save-history", action="store_true", help="Save chat history if enabled"
    )
    return parser


def main(model_name_or_path: str,
         set_url: bool ,
         save_history: bool
         ):
    chatbot = LangchainChatbot(model_name_or_path=model_name_or_path)
    if set_url:
        url = input("Please set your url: ")
        chatbot.set_retriever_url(url)
    while True:
        human_input = input("user: ")
        if human_input == "exit":
            break
        response = chatbot.chat_with_chatbot(human_input)
        print(f"chatbot: {response}")
    if save_history:
        if not os.path.exists("chat_result"):
            os.mkdir("chat_result")
        with open(f"chat_result/{model_name_or_path}.txt", 'w') as file:
            file.write(str(chatbot.memory['abc123'].messages))


if __name__ == "__main__":
    args = get_cli().parse_args()
    main(**vars(args))
