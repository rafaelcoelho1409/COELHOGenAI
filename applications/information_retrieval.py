import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.memory.buffer import ConversationBufferMemory
from functions import (
    InformationRetrieval,
    reload_active_models,
    check_model_and_temperature,
    initialize_shared_memory
)

initialize_shared_memory()

model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()

tools_dict = {
    "Arxiv": "arxiv",
    #"DuckDuckGo": "ddg-search",
    #"LLM Math": "llm-math",
    #"PubMed": "pubmed",
    #"Requests": "requests_all",
    #"Wikipedia": "wikipedia",
    #"Yahoo Finance": "yfinance",
    #"Stack Exchange": "stackexchange"
}
tools_filter = st.sidebar.selectbox(
    label = "Tools",
    options = [None] + list(tools_dict.keys()),
)
if tools_filter == None:
    st.info("You need to select at least one tool.")
    st.stop()

role = InformationRetrieval()
model = role.load_model(
    [tools_dict[tools_filter]], 
    st.session_state["model_name"], 
    st.session_state["temperature_filter"],
    st.session_state["shared_memory"])


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
            prompt,
            #{"input": prompt}, 
            config)
        st.write(response["output"])