import streamlit as st

st.title("langchain!")

prompt = st.chat_input("What's up?")
print(prompt)

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        res = "こんちは"
        st.markdown(res)
