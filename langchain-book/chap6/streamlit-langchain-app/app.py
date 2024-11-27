import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler

from typing import List

load_dotenv()

st.title("langchain!")

def create_agent_chain():
    chat=ChatOpenAI(
        model=os.environ["OPENAI_API_MODEL"],
        temperature=0.5,
        streaming=True,
    )
    tools=load_tools(["ddg-search", "wikipedia"])
    return initialize_agent(
        llm=chat,
        tools=tools,
        agent=AgentType.OPENAI_MULTI_FUNCTIONS
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What's up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        agent_chain=create_agent_chain()
        response=agent_chain.run(prompt, callbacks=[callback])
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
