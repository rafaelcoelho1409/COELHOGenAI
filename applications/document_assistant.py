import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from functions import (
    DocumentAssistant,
    check_model_and_temperature,
    initialize_shared_memory
)

initialize_shared_memory()

model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()

loader_framework = st.sidebar.selectbox(
    label = "Document Loader Framework",
    options = [
        "Docling",
        "LangChain"
    ]
)

role = DocumentAssistant()
model = role.load_model(
    st.session_state["temperature_filter"], 
    st.session_state["model_name"],
    st.session_state["shared_memory"],
    loader_framework
    )


for msg in st.session_state["history"].messages:
    st.chat_message(msg.type).write(msg.content)


if prompt := st.chat_input():
    st.chat_message("human").markdown(prompt)
    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        config = {
            "configurable": {
                "session_id": "any"
                }, 
            "callbacks": [st_callback]}
        response = model.invoke(
            {"input": prompt}, 
            config)
        st.write(response["response"])