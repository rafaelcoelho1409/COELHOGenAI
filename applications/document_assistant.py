import streamlit as st
import json
import os
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

try:
    os.mkdir("qdrant_langchain")
except:
    pass
role = DocumentAssistant(st.session_state["model_name"], "qdrant_langchain")
model = role.load_model(
    st.session_state["temperature_filter"], 
    st.session_state["model_name"],
    st.session_state["shared_memory"],
    loader_framework
    )


COLLECTION_NAME = "docling"
available_filetypes = ["pdf", "jpg", "jpeg", "png", "webp", "docx", "html", "pptx", "adoc", "asciidoc", "md"]
if loader_framework == "Docling":
    with st.sidebar.form("Upload file to analyze"):
        uploaded_file = st.file_uploader(
            "Upload file", 
            type = available_filetypes)
        submit_path = st.form_submit_button(
            label = "Extract",
            use_container_width = True
        )
if submit_path:
    if uploaded_file:
        st.session_state["uploaded_file_content"] = uploaded_file.read()
        st.session_state["uploaded_file_name"] = uploaded_file.name
if not "uploaded_file" in st.session_state:
    st.info("Upload a file to start using Document Assistant.")
    st.stop()

processed_doc = role.process_document(
    st.session_state["uploaded_file_content"], 
    st.session_state["uploaded_file_name"],
    loader_framework)
if loader_framework == "Docling":
    role.save_artifacts_docling(processed_doc)
role.store_on_qdrant(processed_doc, COLLECTION_NAME)
#for x in retrieved_docs:
#    st.write(x)
#st.stop()


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
        #response = model.invoke(
        #    {"input": prompt}, 
        #    config)
        #st.write(response["response"])
        response = role.qdrant_client.query(
            COLLECTION_NAME,
            query_text = prompt,#"what is docling about?",
            limit = 10
        )
        for x in response:
            st.write(x)