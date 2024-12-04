import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from functions import (
    Assistant,
    reload_active_models
)

role = Assistant()
try:
    model = role.load_model(
        st.session_state["temperature_filter"], 
        st.session_state["model_name"]
        )
except:
    st.info("Choose options in settings and click in 'Run model' button to start using COELHO GenAI.")


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
        st.write(response["response"])