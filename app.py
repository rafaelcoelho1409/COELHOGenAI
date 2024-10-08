import streamlit as st
from streamlit_extras.grid import grid
import pandas as pd
import json
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.tools import ShellTool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama.llms import OllamaLLM
from functions import (
    Assistant,
    InformationRetrieval,
    PromptEngineering,
    PDFAssistant,
    SoftwareDevelopment,
    DataScience,
    PlanAndSolve,
    reload_active_models,
    settings,
    prompt_settings,
    prompt_informations,
    image_border_radius
)

st.set_page_config(
    page_title = "COELHO GenAI",
    layout = "wide"
)

#TITLE
#grid_title = grid([5, 1], vertical_align = True)
#container1 = grid_title.container()
#container1.title("$$\\large{\\textbf{ORÁCULO}}$$")
#container1.caption("Author: Rafael Silva Coelho")

#FILTERS
st.sidebar.title("$$\\textbf{COELHO GenAI}$$")
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
    st.info("Choose options in settings and click in 'Run model' button to start using COELHO GenAI.")
    grid_logo = grid([0.15, 0.7, 0.15], vertical_align = True)
    grid_logo.container()
    image_border_radius("assets/coelho_genai_logo.png", 20, 100, 100, grid_logo)
    st.stop()
role_filter = st.session_state["role_filter"]
temperature_filter = st.session_state["temperature_filter"]
with st.sidebar.expander("**Informations**", expanded = True):
    st.markdown(f"**Role:** {role_filter}")
    st.markdown(f"**Model:** {models_filter}")
    st.markdown(f"**Temperature:** {temperature_filter}")
    reload_active_models()
st.sidebar.divider()


if role_filter == "Assistant":
    role = Assistant()
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
        #"Arxiv": "arxiv",
        "DuckDuckGo": "ddg-search",
        "LLM Math": "llm-math",
        "PubMed": "pubmed",
        "Requests": "requests_all",
        "Wikipedia": "wikipedia",
        #"Yahoo Finance": "yfinance",
        #"Stack Exchange": "stackexchange"
    }
    tools_filter = st.sidebar.multiselect(
        label = "Tools",
        options = tools_dict.keys(),
    )
    if tools_filter == []:
        st.info("You need to select at least one tool.")
        st.stop()
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
    role = PlanAndSolve()
    model = role.load_model(models_filter, temperature_filter)

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
    elif role_filter == "Plan And Solve":
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
if "model_memory" not in st.session_state:
    st.session_state["model_memory"] = role.memory


if role_filter == "Prompt Engineering":
    st.sidebar.markdown(f"**Prompt name:** {PROMPT_NAME}")
    #create a memory persistence in an external JSON
    #All code below creates a persistent memory in JSON that gathers all
    #input variables and pass it as argument to model.invoke() method.
    try:
        with open("input_variables_temp.json", "r") as file:
            input_var_temp = json.load(file)
    except FileNotFoundError:
        data = {}
        with open("input_variables_temp.json", "w") as file:
            json.dump(data, file)
    with open("input_variables_temp.json", "r") as file:
        data_temp = json.load(file)
    if type(PROMPT) in [ChatPromptTemplate, StructuredPrompt]:
        #SYSTEM INPUT VARIABLES
        if role.system_input_variables != []:
            for i, input_variable in enumerate(role.system_input_variables[0]):
                globals()[f"prompt_{input_variable}"] = st.sidebar.text_area(
                    label = input_variable
                )
                if globals()[f"prompt_{input_variable}"] is not None:
                    data_temp[input_variable] = globals()[f"prompt_{input_variable}"]
                    with open("input_variables_temp.json", "w") as file:
                        json.dump(data_temp, file)
        #HUMAN INPUT VARIABLES
        if role.human_input_variables != []:
            input_variable_info = ""
            for i, input_variable in enumerate(role.human_input_variables[0]):
                globals()[f"prompt_{input_variable}"] = st.chat_input(
                    placeholder = input_variable
                )
                if globals()[f"prompt_{input_variable}"] is not None:
                    data_temp[input_variable] = globals()[f"prompt_{input_variable}"]
                    input_variable_info += f"**{input_variable}:** {data_temp[input_variable]}"
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
    prompt_info = ""
    for key in role.input_variables:
        try:
            prompt_info += f"**{key}:** {data_temp[key]}\n\n"
        except:
            pass
    st.info(prompt_info)
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
            try:
                response = model.invoke(
                    data_temp,
                    config
                )
            except:
                pass
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
            try:
                response["text"] = response["response"]
            except:
                try:
                    response["text"] = response["output"]
                except:
                    pass
                    try:
                        st.write(response["text"])
                    except:
                        try:
                            st.write(response)
                        except:
                            pass
                        #st.error("Error")
            #st.markdown(response["text"])