import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from functions import (
    InformationRetrieval,
    reload_active_models
)


tools_dict = {
    #"Arxiv": "arxiv",
    #"DuckDuckGo": "ddg-search",
    "LLM Math": "llm-math",
    "PubMed": "pubmed",
    "Requests": "requests_all",
    "Wikipedia": "wikipedia",
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
    st.session_state["temperature_filter"])


with st.sidebar.expander("**Informations**", expanded = True):
    st.markdown(f"**Model:** {st.session_state["model_name"]}")
    st.markdown(f"**Temperature:** {st.session_state["temperature_filter"]}")
    reload_active_models()


for msg in role.history.messages:
    st.chat_message(msg.type).write(msg.content)
#if "role" not in st.session_state:
st.session_state["role"] = role
#if "model_memory" not in st.session_state:
st.session_state["model_memory"] = role.memory


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
        st.write(response["output"])