import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.tools import Tool
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools.shell.tool import ShellTool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.tools.pubmed.tool import PubmedQueryRun
#from langchain_community.utilities.searx_search import SearxSearchWrapper
from langchain_community.utilities.stackexchange import StackExchangeAPIWrapper
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun, WikipediaAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools.youtube.search import YouTubeSearchTool
from langchain_experimental.utilities.python import PythonREPL
from functions import (
    InformationRetrieval,
    check_model_and_temperature,
    initialize_shared_memory
)

initialize_shared_memory()

model_temperature_checker = check_model_and_temperature()
if model_temperature_checker == False:
    st.info("Choose model and temperature to start running COELHO GenAI models.")
    st.stop()

tools_dict = {
    "Arxiv": ArxivAPIWrapper(),
    "Shell": ShellTool(),
    "DuckDuckGo": DuckDuckGoSearchResults(),
    "Python": PythonREPL(),
    "PubMed": PubmedQueryRun(),
    #"SearxNG": SearxSearchWrapper(searx_host = "http://localhost:8888"),
    "StackExchange": StackExchangeAPIWrapper(),
    "Wikidata": WikidataQueryRun(api_wrapper = WikidataAPIWrapper()),
    "Wikipedia": WikipediaQueryRun(api_wrapper = WikipediaAPIWrapper()),
    "Yahoo Finance News": YahooFinanceNewsTool(),
    #"YouTube Search": YouTubeSearchTool()
}
tools_filter = st.sidebar.selectbox(
    label = "Tools",
    options = [None] + list(tools_dict.keys()),
)
if tools_filter == None:
    st.info("You need to select at least one tool.")
    st.stop()

role = InformationRetrieval()
model = role.load_model(
    st.session_state["framework"],
    [
        Tool(
            tools_dict[tools_filter].__class__.__name__,
            func = tools_dict[tools_filter].run,
            description = tools_dict[tools_filter].__class__.__name__,
        )
    ],
    st.session_state["model_name"], 
    st.session_state["temperature_filter"],
    st.session_state["shared_memory"])


for msg in st.session_state["history"].messages:
    st.chat_message(msg.type).write(msg.content)


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
            prompt,
            #{"input": prompt}, 
            config)
        st.write(response["output"])