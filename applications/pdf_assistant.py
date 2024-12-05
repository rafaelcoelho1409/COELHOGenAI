import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_ollama.llms import OllamaLLM
from functions import (
    PDFAssistant,
    reload_active_models
)


with st.sidebar.form("PDFAssistant"):
    uploaded_file = st.file_uploader(
        'Upload file', 
        type = 'pdf'
    )
    send_files_button = st.form_submit_button(
        label = "Send file",
        use_container_width = True
    )
if uploaded_file is not None:
    st.sidebar.success("File uploaded with success!")
    llm = OllamaLLM(
        model = st.session_state["model_name"],
        temperature = st.session_state["temperature_filter"])
    role = PDFAssistant(llm, st.session_state["model_name"], uploaded_file)
    #if send_files_button:
    with st.spinner("Reading PDF"):
        raw_text = role.pdf_read(uploaded_file)
    with st.spinner("Getting chunks"):
        text_chunks = role.get_chunks(raw_text)
    with st.spinner("Storing vectors"):
        role.vector_store(text_chunks)


with st.sidebar.expander("**Informations**", expanded = True):
    st.markdown(f"**Model:** {st.session_state["model_name"]}")
    st.markdown(f"**Temperature:** {st.session_state["temperature_filter"]}")
    reload_active_models()


for msg in role.history.messages:
    st.chat_message(msg.type).write(msg.content)
st.session_state["role"] = role
st.session_state["model_memory"] = role.memory


if prompt := st.chat_input():
    #reload_active_models()
    st.chat_message("human").markdown(prompt)
    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        config = {"configurable": {"session_id": "any"}, "callbacks": [st_callback]}
        retrieval_chain = st.session_state["role"].process_user_input(prompt)
        model = st.session_state["role"].load_model(retrieval_chain)
        try:
            response = model.invoke(
                {"input": prompt}, 
                config)
        except:
            response = model.run(prompt)
        #response
        st.write(response)

#IMPORTANT: MAKE CORRECTIONS IN CODE BECAUSE THERE'S NO ROLE FUNCTION IN PDFAssistant CLASS