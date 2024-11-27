import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

load_dotenv()

def create_agent_chain():
    chat=ChatOpenAI(
        model=os.environ["OPENAI_API_MODEL"],
        temperature=0.5,
        streaming=True,
    )
    agent_kwargs={
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }

    memory=ConversationBufferMemory(memory_key="memory", return_messages=True)

    tools=load_tools(["ddg-search", "wikipedia"])
    return initialize_agent(
        llm=chat,
        tools=tools,
        agent=AgentType.OPENAI_MULTI_FUNCTIONS,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )

if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()

st.title("langchain!")


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
        response=st.session_state.agent_chain.run(prompt, callbacks=[callback])
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
