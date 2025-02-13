import streamlit as st
from streamlit_extras.grid import grid
from functions import (
    settings,
    check_model_and_temperature,
    initialize_shared_memory
)

st.set_page_config(
    page_title = "COELHO GenAI", 
    page_icon = ":material/home:",
    layout = "wide")

home = st.Page(
    "applications/home.py", 
    title = "Home", 
    icon = ":material/home:")
assistant = st.Page(
    "applications/assistant.py", 
    title = "Assistant", 
    icon = ":material/edit:")
information_retrieval = st.Page(
    "applications/information_retrieval.py", 
    title = "Information Retrieval", 
    icon = ":material/edit:")
data_science = st.Page(
    "applications/data_science.py", 
    title = "Data Science", 
    icon = ":material/edit:")
#prompt_engineering = st.Page(
#    "applications/prompt_engineering.py", 
#    title = "Prompt Engineering", 
#    icon = ":material/edit:")
software_development = st.Page(
    "applications/software_development.py", 
    title = "Software Development", 
    icon = ":material/edit:")
plan_and_solve = st.Page(
    "applications/plan_and_solve.py", 
    title = "Plan & Solve", 
    icon = ":material/edit:")
document_assistant = st.Page(
    "applications/document_assistant.py", 
    title = "Document Assistant", 
    icon = ":material/edit:")


pg = st.navigation({
    "COELHO GenAI by Rafael Coelho": [
        home],
    "Applications": [
        assistant,
        information_retrieval,
        data_science,
        #prompt_engineering,
        document_assistant,
        software_development,
        plan_and_solve,
    ]
})

with open("style.css") as css:
    st.html(f"<style>{css.read()}</style>")

with st.container(key = "app_title"):
    st.title(("$$\\textbf{" + pg.title + "}$$").replace("&", "\&"))

grid_buttons = st.sidebar.columns(2)
settings_button = grid_buttons[0].button(
    label = "Settings",
    use_container_width = True
)
if settings_button:
    settings()
clear_memory_button = grid_buttons[1].button(
    label = "Clear memory",
    use_container_width = True
)    


pg.run()


model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    if pg.title != "Home":
        st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()


with st.sidebar.expander("**Informations**", expanded = True):
    st.markdown(f"**Framework:** {st.session_state['framework']}")
    st.markdown(f"**Model:** {st.session_state["model_name"]}")
    st.markdown(f"**Temperature:** {st.session_state["temperature_filter"]}")


initialize_shared_memory()
if clear_memory_button:
    st.session_state["shared_memory"].clear()