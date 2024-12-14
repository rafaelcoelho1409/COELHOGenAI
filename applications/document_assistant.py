import streamlit as st
import json
import os
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from functions import (
    DocumentAssistant,
    check_model_and_temperature,
    initialize_shared_memory,
    docling_process_document,
    docling_save_artifacts,
    docling_store_on_qdrant
)

initialize_shared_memory()

model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()


loaders_filters = st.sidebar.container()
loaders_filters_grid = loaders_filters.columns(2)
loader_framework = loaders_filters_grid[0].selectbox(
    label = "Document Loader",
    options = [
        "Docling",
        #"LangChain"
    ]
)
if loader_framework == "Docling":
    docling_type = loaders_filters_grid[1].selectbox(
        label = "Type",
        options = [
            "File",
            "URL"
        ]
    )

try:
    os.mkdir("qdrant_langchain")
except:
    pass
role = DocumentAssistant(
    st.session_state["model_name"], 
    #"qdrant_langchain"
    )
model = role.load_model(
    st.session_state["temperature_filter"], 
    st.session_state["model_name"],
    st.session_state["shared_memory"],
    loader_framework
    )


COLLECTION_NAME = "docling"
available_filetypes = ["pdf", "jpg", "jpeg", "png", "webp", "docx", "html", "pptx", "adoc", "asciidoc", "md"]
if loader_framework == "Docling":
    if docling_type == "File":
        with st.sidebar.form("Upload file to analyze"):
            uploaded_file = st.file_uploader(
                "Upload file", 
                type = available_filetypes)
            submit = st.form_submit_button(
                label = "Extract",
                use_container_width = True
            )
    elif docling_type == "URL":
        with st.sidebar.form("Set a URL to analyze"):
            url = st.text_input(
                label = "URL"
            )
            submit = st.form_submit_button(
                label = "Extract",
                use_container_width = True
            )
    if submit:
        if docling_type == "File":
            if uploaded_file:
                st.session_state["uploaded_file_content"] = uploaded_file.read()
                st.session_state["uploaded_file_name"] = uploaded_file.name
        elif docling_type == "URL":
            st.session_state["url"] = url
    if docling_type == "File":
        if not "uploaded_file_content" in st.session_state:
            st.info("Upload a file to start using Document Assistant.")
            st.stop()
        processed_doc = docling_process_document(
            docling_type,
            uploaded_file_content = st.session_state["uploaded_file_content"], 
            uploaded_file_name = st.session_state["uploaded_file_name"]
            )
    elif docling_type == "URL":
        if not "url" in st.session_state:
            st.info("Set a URL to start using Document Assistant.")
            st.stop()
        processed_doc = docling_process_document(
            docling_type,
            url = st.session_state["url"]
        )
    docling_save_artifacts(processed_doc)
    vector_store = docling_store_on_qdrant(
        role.qdrant_client,
        processed_doc, 
        COLLECTION_NAME, 
        st.session_state["model_name"])


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
        rag_query = vector_store.similarity_search(query = prompt, k = 3)
        rag_result = "\n\n".join(x.page_content for x in rag_query)
        response = model.invoke(
            {
                "context": rag_result,
                "input": prompt
                }, 
            config)
        st.write(response["text"])