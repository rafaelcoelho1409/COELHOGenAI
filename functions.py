import streamlit as st
import base64
import ollama
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory.buffer import ConversationBufferMemory
from langchain_ollama.chat_models import ChatOllama
from langchain.chains.conversation.base import ConversationChain

#>>>-------------------------------------------------<<<
#STREAMLIT
#>>>-------------------------------------------------<<<
@st.dialog("Settings")
def settings():
    with st.form("Settings"):
        models_options = sorted(
            [x["model"] for x in ollama.list()["models"]])
        if ollama.ps()["models"] != []:
            active_models = [x["model"] for x in ollama.ps()["models"]]
            models_filter = st.selectbox(
                label = "Ollama Models",
                options = models_options,
                index = models_options.index(active_models[0])
            )
        else:
            try:
                #try to get the last model used, if exists
                models_filter = st.selectbox(
                    label = "Ollama Models",
                    options = models_options,
                    index = models_options.index(st.session_state["model_name"])
                )
            except:
                models_filter = st.selectbox(
                    label = "Ollama Models",
                    options = sorted([x["model"] for x in ollama.list()["models"]])
                )
        temperature_filter = st.slider(
            label = "Temperature",
            min_value = 0.00,
            max_value = 1.00,
            value = 0.00,
            step = 0.01
        )
        submit_button = st.form_submit_button(
                label = "Run model",
                use_container_width = True
            )
        if submit_button:
            st.session_state["model_name"] = models_filter
            st.session_state["temperature_filter"] = temperature_filter
            st.rerun()


#>>>-------------------------------------------------<<<
#FUNCTIONS
#>>>-------------------------------------------------<<<
def image_border_radius(image_path, border_radius, width, height, page_object = None, is_html = False):
    if is_html == False:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        # Create HTML string with the image
        img_html = f'<img src="data:image/jpeg;base64,{img_base64}" style="border-radius: {border_radius}px; width: {width}%; height: {height}%">'
        # Display the HTML string in Streamlit
        if page_object == None:
            st.markdown(img_html, unsafe_allow_html=True)
        else:
            page_object.markdown(img_html, unsafe_allow_html=True)
    else:
        # Create HTML string with the image
        img_html = f'<img src="{image_path}" style="border-radius: {border_radius}px; width: 300px;">'
        # Display the HTML string in Streamlit
        if page_object == None:
            st.markdown(img_html, unsafe_allow_html=True)
        else:
            page_object.markdown(img_html, unsafe_allow_html=True)

def reload_active_models():
    active_models_container = st.container()
    active_models_text = "## Active models (Ollama)\n"
    if ollama.ps()["models"] != []:
        for model_name in ollama.ps()["models"]:
            active_models_text += f"* {model_name['model']}\n"
    else:
        active_models_text += "No active models."
    active_models_container.info(active_models_text)
#>>>-------------------------------------------------<<<
#CLASSES
#>>>-------------------------------------------------<<<
class Assistant:
    def __init__(self):
        self.history = StreamlitChatMessageHistory(key = "chat_history")
        self.prompt_template = """
            You are a nice chatbot having a conversation with a human.
    
            Chat history:
            {chat_history}
    
            Human: {input}
            """
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)
        self.memory = ConversationBufferMemory(
            memory_key = "chat_history", 
            return_messages = True,
            chat_memory = self.history)
    def load_model(self, temperature_filter, model_name):
        llm = ChatOllama(
                model = model_name, 
                temperature = temperature_filter)
        conversation = ConversationChain(
            llm = llm,
            prompt = self.prompt,
            verbose = True,
            memory = self.memory
        )
        return conversation