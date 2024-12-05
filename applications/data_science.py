import streamlit as st
import pandas as pd
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from functions import (
    DataScience,
    reload_active_models
)


ds_framework = st.sidebar.selectbox(
    label = "Framework",
    options = [
        "LangChain",
        "PandasAI"
    ]
)
st.session_state["ds_framework"] = ds_framework
role = DataScience(ds_framework)
with st.sidebar.form("DataScience"):
    uploaded_file = st.file_uploader(
        'Upload file', 
        type = ['csv', 'xlsx']
    )
    send_files_button = st.form_submit_button(
        label = "Send file",
        use_container_width = True
    )
    if uploaded_file is not None:
        st.sidebar.success("File uploaded with success!")
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension == 'xlsx':
            df = pd.read_excel(uploaded_file, engine = 'openpyxl')
        #---
        model = role.load_model(
            df, 
            st.session_state["model_name"], 
            st.session_state["temperature_filter"])


with st.sidebar.expander("**Informations**", expanded = True):
    st.markdown(f"**Model:** {st.session_state["model_name"]}")
    st.markdown(f"**Temperature:** {st.session_state["temperature_filter"]}")
    reload_active_models()


for msg in role.history.messages:
    st.chat_message(msg.type).write(msg.content)
st.session_state["role"] = role
st.session_state["model_memory"] = role.memory


if uploaded_file is None:
    st.info("Upload a data file to use the Data Science assistant.")
    st.stop()


if prompt := st.chat_input():
    #reload_active_models()
    st.chat_message("human").markdown(prompt)
    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        config = {
            "configurable": {
                "session_id": "any"
            }, 
            "callbacks": [st_callback]}
        if st.session_state["ds_framework"] == "PandasAI":
            response = model.chat(prompt)
            try:    
                st.dataframe(response)
            except:
                st.write(response)
            st.stop()
        else:
            try:
                response = model.invoke(
                    {"input": prompt}, 
                    config)
            except:
                response = model.run(prompt)
        #response
        st.write(response)