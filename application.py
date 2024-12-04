import streamlit as st
from functions import (
    settings,
    reload_active_models
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
prompt_engineering = st.Page(
    "applications/prompt_engineering.py", 
    title = "Prompt Engineering", 
    icon = ":material/edit:")
pdf_assistant = st.Page(
    "applications/pdf_assistant.py", 
    title = "PDF Assistant", 
    icon = ":material/edit:")
software_development = st.Page(
    "applications/software_development.py", 
    title = "Software Development", 
    icon = ":material/edit:")
plan_and_solve = st.Page(
    "applications/plan_and_solve.py", 
    title = "Plan & Solve", 
    icon = ":material/edit:")

pg = st.navigation({
    "COELHO GenAI by Rafael Coelho": [
        home],
    "Applications": [
        assistant,
        information_retrieval,
        data_science,
        prompt_engineering,
        pdf_assistant,
        software_development,
        plan_and_solve
    ]
})


st.sidebar.title(pg.title)
settings_button = st.sidebar.button(
    label = "Settings",
    use_container_width = True
)
if settings_button:
    settings()
    reload_active_models()

pg.run()