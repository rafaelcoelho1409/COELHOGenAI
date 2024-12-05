import streamlit as st
import json
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts.structured import StructuredPrompt
from functions import (
    PromptEngineering,
    reload_active_models,
    prompt_settings,
    prompt_informations
)


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
    st.session_state["temperature_filter"])
model = RunnableWithMessageHistory(
    model,
    lambda session_id: role.history,  # Always return the instance created earlier
    input_messages_key = role.input_variables,
    history_messages_key = "chat_history",
)


with st.sidebar.expander("**Informations**", expanded = True):
    st.markdown(f"**Model:** {st.session_state["model_name"]}")
    st.markdown(f"**Temperature:** {st.session_state["temperature_filter"]}")
    reload_active_models()


for msg in role.history.messages:
    st.chat_message(msg.type).write(msg.content)
st.session_state["role"] = role
st.session_state["model_memory"] = role.memory


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
        st.write(response)