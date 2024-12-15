import streamlit as st
import inspect
import sys
import subprocess
import re
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community import document_loaders
from functions import (
    DocumentAssistant,
    check_model_and_temperature,
    initialize_shared_memory,
    docling_process_document,
    docling_save_artifacts,
    store_on_qdrant,
    retrieved_documents
)

initialize_shared_memory()

model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()


loader_framework = st.sidebar.selectbox(
    label = "Document Loader",
    options = [
        "Docling",
        "LangChain"
    ]
)


role = DocumentAssistant(
    st.session_state["model_name"],
    st.session_state["vector_database_filter"]
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
    docling_type = st.sidebar.selectbox(
        label = "Type",
        options = [
            "File",
            "URL"
        ]
    )
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
            if url == "":
                st.error("Use a valid URL to start using Document Assistant.")
                st.stop()
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
    #docling_save_artifacts(processed_doc)
    if st.session_state["rag_filter"] == True:
        st.session_state["vector_store"] = store_on_qdrant(
            role.qdrant_client,
            processed_doc, 
            st.session_state["model_name"],
            loader_framework)
elif loader_framework == "LangChain":
    langchain_loader_type = st.sidebar.selectbox(
        label = "Type",
        options = document_loaders.__all__,
        index = document_loaders.__all__.index("WikipediaLoader")
    )
    st.sidebar.caption("LangChain Document Loaders (Experimental)")
    loader = document_loaders.__getattr__(langchain_loader_type)
    args_empty = {
        name: param.default
        for name, param in inspect.signature(loader).parameters.items()
        if param.default is param.empty
    }
    args_not_empty = {
        name: param.default
        for name, param in inspect.signature(loader).parameters.items()
        if param.default is not param.empty
    }
    with st.sidebar.form(langchain_loader_type):
        st.subheader(langchain_loader_type)
        for k, v in args_empty.items():
            globals()[f"{langchain_loader_type}__{k}"] = st.text_input(
                label = k,
                value = "" if v is inspect._empty else v
            )
        st.divider()
        for k, v in args_not_empty.items():
            globals()[f"{langchain_loader_type}__{k}"] = st.text_input(
                label = k,
                value = v
            )
        submit_args = st.form_submit_button(
            "Submit",
            use_container_width = True)
    if submit_args:
        loader_args = {
            k: globals()[f"{langchain_loader_type}__{k}"] for k in args_empty.keys()
            } | {
            k: globals()[f"{langchain_loader_type}__{k}"] for k in args_not_empty.keys()}
        try:
            st.session_state["langchain_processed_doc"] = loader(**loader_args).load()
            if st.session_state["rag_filter"] == True:
                st.session_state["vector_store"] = store_on_qdrant(
                    role.qdrant_client,
                    st.session_state["langchain_processed_doc"], 
                    st.session_state["model_name"],
                    loader_framework)
        except ImportError as e:
            match = re.search(r'pip install\s+([^\s]+)', str(e))
            if match:
                package_name = match.group(1).replace(r"`", "").replace(r".", "")
                with st.spinner(f"Downloading library: {package_name}"):
                    test = subprocess.check_call([
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        package_name
                    ],
                    )
                st.rerun()
                st.info("Click in submit again to rerun the tool.")
        except Exception as e:
            st.error(e)
            st.stop()

try:
    view_retrieved_documents = st.sidebar.button(
        label = "View retrieved documents",
        use_container_width = True,
    )
    if view_retrieved_documents:
        if loader_framework == "Docling":
            retrieved_documents(processed_doc, loader_framework)
        elif loader_framework == "LangChain":
            retrieved_documents(st.session_state["langchain_processed_doc"], loader_framework)
except:
    pass

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
        if st.session_state["memory_filter"] == False:
            st.session_state["shared_memory"].clear()
        if st.session_state["rag_filter"] == True:
            rag_query = st.session_state["vector_store"].similarity_search(query = prompt, k = 10)
            rag_result = "\n\n".join(x.page_content for x in rag_query)
        else:
            if loader_framework == "Docling":
                rag_result = processed_doc.document.export_to_markdown()
            elif loader_framework == "LangChain":
                rag_result = "\n\n".join(
                    x.page_content for x in st.session_state["langchain_processed_doc"])
        response = model.invoke(
            {
                "context": rag_result,
                "input": prompt
                }, 
            config)
        st.write(response["text"])