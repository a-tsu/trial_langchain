# https://qiita.com/hideki/items/d3a474c85cdb7eb8e936

import os

openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
system_template = """以下は、HumanとAIが仲良く会話している様子です。AIは饒舌で、その文脈から具体的な内容をたくさん教えてくれます。AIは質問に対する答えを知らない場合、正直に「知らない」と答えます。"""
system_prompt = SystemMessagePromptTemplate.from_template(
    system_template
)
memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=100, return_messages=True
 )
prompts = ChatPromptTemplate.from_messages([
    system_prompt,
    MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
])
chain = ConversationChain(
    llm=llm, prompt=prompts, memory=memory, verbose=True
)

mesg = chain.run("こんにちは")
print(mesg)

