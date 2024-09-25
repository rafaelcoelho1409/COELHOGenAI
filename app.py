import streamlit as st
from streamlit_extras.grid import grid
import pandas as pd
import json
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama.llms import OllamaLLM
from functions import (
    Oraculo,
    InformationRetrieval,
    PromptEngineering,
    PDFAssistant,
    SoftwareDevelopment,
    DataScience,
    PlanAndSolve,
    reload_active_models,
    settings,
    prompt_settings,
    prompt_informations
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
with st.sidebar.expander("**Informations**", expanded = True):
    st.markdown(f"**Role:** {role_filter}")
    st.markdown(f"**Model:** {models_filter}")
    st.markdown(f"**Temperature:** {temperature_filter}")
    reload_active_models()
st.sidebar.divider()


if role_filter == "Oráculo":
    role = Oraculo()
    model = role.load_model(
        temperature_filter, 
        st.session_state["model_name"]
        )
if role_filter == "Prompt Engineering":
    prompt_settings_button = st.sidebar.button(
        label = "Prompt settings",
        use_container_width = True
    )
    if prompt_settings_button:
        prompt_settings()
    try:
        PROMPT_NAME = st.session_state["PROMPT_NAME"]
        PROMPT = st.session_state["PROMPT"]
    except:
        st.info("Load a prompt from LangChain Hub to start.")
        st.stop()
    prompt_informations_button = st.sidebar.button(
        label = "Prompt informations",
        use_container_width = True
    )
    if prompt_informations_button:
        prompt_informations(PROMPT_NAME, PROMPT)
    role = PromptEngineering(PROMPT)
    model = role.load_model(
        st.session_state["model_name"], 
        temperature_filter)
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

try:
    if (
        role_filter == "Data Science" 
        and st.session_state["ds_framework"] == "PandasAI"):
        pass
    elif role_filter == "Prompt Engineering":
        model = RunnableWithMessageHistory(
            model,
            lambda session_id: role.history,  # Always return the instance created earlier
            input_messages_key = role.input_variables,
            history_messages_key = "chat_history",
        )        
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
if "model_memory" not in st.session_state:
    st.session_state["model_memory"] = role.memory


if role_filter == "Prompt Engineering":
    st.sidebar.markdown(f"**Prompt name:** {PROMPT_NAME}")
    #create a memory persistence in an external JSON
    try:
        with open("input_variables_temp.json", "r") as file:
            input_var_temp = json.load(file)
    except FileNotFoundError:
        data = {}
        with open("input_variables_temp.json", "w") as file:
            json.dump(data, file)
    with open("input_variables_temp.json", "r") as file:
        data_temp = json.load(file)
    if type(PROMPT) == ChatPromptTemplate:
        #SYSTEM INPUT VARIABLES
        for i, input_variable in enumerate(role.system_input_variables[0]):
            globals()[f"prompt_{input_variable}"] = st.sidebar.text_area(
                label = input_variable
            )
            if globals()[f"prompt_{input_variable}"] is not None:
                data_temp[input_variable] = globals()[f"prompt_{input_variable}"]
                with open("input_variables_temp.json", "w") as file:
                    json.dump(data_temp, file)
        #HUMAN INPUT VARIABLES
        for i, input_variable in enumerate(role.human_input_variables[0]):
            globals()[f"prompt_{input_variable}"] = st.chat_input(
                placeholder = input_variable
            )
            if globals()[f"prompt_{input_variable}"] is not None:
                data_temp[input_variable] = globals()[f"prompt_{input_variable}"]
                with open("input_variables_temp.json", "w") as file:
                    json.dump(data_temp, file)
    elif type(PROMPT) == PromptTemplate:
        for i, input_variable in enumerate(role.input_variables):
            globals()[f"prompt_{input_variable}"] = st.chat_input(
                placeholder = input_variable
            )
            if globals()[f"prompt_{input_variable}"] is not None:
                data_temp[input_variable] = globals()[f"prompt_{input_variable}"]
                with open("input_variables_temp.json", "w") as file:
                    json.dump(data_temp, file)
    #read stored data
    with open("input_variables_temp.json", "r") as file:
        data_temp = json.load(file)
    if sorted(list(data_temp.keys())) == sorted(role.input_variables):
        prompt_set = ""
        for key in role.input_variables:
            prompt_set += f"**{key}:** {data_temp[key]}\n\n"
        st.chat_message("human").markdown(prompt_set)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            config = {"configurable": {"session_id": "any"}, "callbacks": [st_callback]}
            #deleting temporary data
            with open("input_variables_temp.json", "w") as file:
                json.dump({}, file)
            response = model.invoke(
                data_temp,
                config
            )
else:
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
                try:
                    response["text"] = response["output"]
                except:
                    pass
            st.markdown(response["text"])