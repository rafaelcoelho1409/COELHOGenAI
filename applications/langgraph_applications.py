import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from functions import (
    LangGraphBasicChatbot,
    LangGraphWikipediaChatbot,
    check_model_and_temperature,
    initialize_shared_memory,
    view_application_graph
)

initialize_shared_memory()

model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()

langgraph_tasks = st.sidebar.selectbox(
    label = "LangGraph Tasks",
    options = [
        "Basic Chatbot",
        "Wikipedia Chatbot"
    ]
)

role_dict = {
    "Basic Chatbot": LangGraphBasicChatbot,
    "Wikipedia Chatbot": LangGraphWikipediaChatbot
}

role = role_dict[langgraph_tasks](
    st.session_state["model_name"],
    st.session_state["temperature_filter"], 
    st.session_state["shared_memory"],
    st.session_state["langgraph_memory"]
)


view_application_graph_button = st.sidebar.button(
    label = "View application graph",
    use_container_width = True
)
if view_application_graph_button:
    view_application_graph(role.graph)


for msg in st.session_state["history"].messages:
    st.chat_message(msg.type).write(msg.content)
    
    
if prompt := st.chat_input():
    role.stream_graph_updates(prompt)