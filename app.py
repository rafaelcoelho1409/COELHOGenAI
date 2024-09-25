import streamlit as st
from streamlit_extras.grid import grid
import ollama
import pandas as pd
from pandasai import SmartDataframe
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.llms import OllamaLLM
from functions import (
    Oraculo,
    InformationRetrieval,
    PDFAssistant,
    SoftwareDevelopment,
    DataScience,
    PlanAndSolve,
    reload_active_models,
    settings
)

st.set_page_config(
    page_title = "Oráculo",
    layout = "wide"
)

#TITLE
#grid_title = grid([5, 1], vertical_align = True)
#container1 = grid_title.container()
#container1.title("$$\\large{\\textbf{ORÁCULO}}$$")
#container1.caption("Author: Rafael Silva Coelho")

#FILTERS
st.sidebar.title("$$\\Large{\\textbf{ORÁCULO}}$$")
st.sidebar.caption("Author: Rafael Silva Coelho")
settings_button = st.sidebar.button(
    label = "Settings",
    use_container_width = True
)
if settings_button:
    settings()

try:
    models_filter = st.session_state["model_name"]
except:
    st.info("Choose options in settings and click in 'Run model' button to start using Oráculo.")
    st.stop()
role_filter = st.session_state["role_filter"]
temperature_filter = st.session_state["temperature_filter"]
st.sidebar.markdown(f"**Role:** {role_filter}")
st.sidebar.markdown(f"**Model:** {models_filter}")
st.sidebar.markdown(f"**Temperature:** {temperature_filter}")
st.sidebar.divider()


if role_filter == "Oráculo":
    role = Oraculo()
    model = role.load_model(
        temperature_filter, 
        st.session_state["model_name"]
        )
elif role_filter == "Information Retrieval":
    tools_dict = {
        "Arxiv": "arxiv",
        "DuckDuckGo": "ddg-search",
        "LLM Math": "llm-math",
        "PubMed": "pubmed",
        "Requests": "requests_all",
        #"Wikidata": "wikidata",
        "Wikipedia": "wikipedia",
        #"Yahoo Finance": "yfinance",
        #"Stack Exchange": "stackexchange"
    }
    tools_filter = st.sidebar.multiselect(
        label = "Tools",
        options = tools_dict.keys(),
    )
    tools = [tools_dict[tool_name] for tool_name in tools_filter]
    st.sidebar.markdown(f"**Tools:** {', '.join(tools_filter)}")
    st.session_state["tools_filter"] = tools_filter
    st.session_state["tools"] = tools
    role = InformationRetrieval()
    model = role.load_model(
        tools, 
        st.session_state["model_name"], 
        temperature_filter)
elif role_filter == "PDF Assistant":
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
            model = models_filter,
            temperature = temperature_filter)
        role = PDFAssistant(llm, models_filter, uploaded_file)
        #if send_files_button:
        with st.spinner("Reading PDF"):
            raw_text = role.pdf_read(uploaded_file)
        with st.spinner("Getting chunks"):
            text_chunks = role.get_chunks(raw_text)
        with st.spinner("Storing vectors"):
            role.vector_store(text_chunks)
elif role_filter == "Software Development":
    role = SoftwareDevelopment()
    model = role.load_model(models_filter, temperature_filter)
elif role_filter == "Data Science":
    ds_framework = st.sidebar.selectbox(
        label = "Framework",
        options = [
            "LangChain",
            "PandasAI"
        ]
    )
    st.session_state["ds_framework"] = ds_framework
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
            role = DataScience(ds_framework)
            model = role.load_model(df, models_filter, temperature_filter)
elif role_filter == "Plan And Solve":
    tools_dict = {
        "Arxiv": "arxiv",
        "DuckDuckGo": "ddg-search",
        "LLM Math": "llm-math",
        "PubMed": "pubmed",
        "Requests": "requests_all",
        #"Wikidata": "wikidata",
        "Wikipedia": "wikipedia",
        #"Yahoo Finance": "yfinance",
        #"Stack Exchange": "stackexchange"
    }
    tools_filter = st.sidebar.multiselect(
        label = "Tools",
        options = tools_dict.keys(),
    )
    tools = [tools_dict[tool_name] for tool_name in tools_filter]
    st.sidebar.markdown(f"**Tools:** {', '.join(tools_filter)}")
    st.session_state["tools_filter"] = tools_filter
    st.session_state["tools"] = tools
    role = PlanAndSolve()
    model = role.load_model(tools, models_filter, temperature_filter)

reload_active_models()

try:
    if role_filter == "Data Science" and st.session_state["ds_framework"] == "PandasAI":
        pass
    else:
        model = RunnableWithMessageHistory(
            model,
            lambda session_id: role.history,  # Always return the instance created earlier
            input_messages_key = "input",
            history_messages_key = "chat_history",
        )
except:
    if role_filter == "Data Science":
        st.info("Upload a data file to use the Data Science assistant.")
        st.stop()

for msg in role.history.messages:
    st.chat_message(msg.type).write(msg.content)
if "role" not in st.session_state:
    st.session_state["role"] = role
#if "model" not in st.session_state:
#    st.session_state["model"] = model
if "model_memory" not in st.session_state:
    st.session_state["model_memory"] = role.memory

if prompt := st.chat_input():
    #reload_active_models()
    st.chat_message("human").markdown(prompt)
    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        config = {"configurable": {"session_id": "any"}, "callbacks": [st_callback]}
        if role_filter == "PDF Assistant":
            retrieval_chain = st.session_state["role"].process_user_input(prompt)
            model = st.session_state["role"].load_model(retrieval_chain)
        elif role_filter == "Data Science" and st.session_state["ds_framework"] == "PandasAI":
            response = model.chat(prompt)
        else:
            response = model.invoke(
                {"input": prompt}, 
                config)
        try:
            response["text"] = response["response"]
        except:
            pass
        st.markdown(response["text"])