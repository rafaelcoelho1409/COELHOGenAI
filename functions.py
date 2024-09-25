import streamlit as st
import ollama
import os
from pandasai import SmartDataframe
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.plan_and_execute import (
    load_chat_planner,
    load_agent_executor,
    PlanAndExecute
)
from langchain_community.tools import ShellTool
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import (
    AgentExecutor, 
    AgentType, 
    initialize_agent,
    create_tool_calling_agent
)
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chains.conversation.base import ConversationChain
from langchain.tools.retriever import create_retriever_tool

def reload_active_models():
    active_models_container = st.sidebar.container()
    active_models_text = "## Active models (Ollama)\n"
    if ollama.ps()["models"] != []:
        for model_name in ollama.ps()["models"]:
            active_models_text += f"* {model_name['name']}\n"
    else:
        active_models_text += "No active models."
    active_models_container.info(active_models_text)


@st.dialog("Settings")
def settings():
    with st.form("Settings"):
        models_options = sorted(
            [x["name"] for x in ollama.list()["models"]])
        role_filter = st.selectbox(
            label = "Role",
            options = [
                "Oráculo",
                "Information Retrieval",
                #"PDF Assistant",
                #"Software Development",
                "Data Science",
                "Plan And Solve"
            ],
        )
        if ollama.ps()["models"] != []:
            active_models = [x["name"] for x in ollama.ps()["models"]]
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
                    options = sorted([x["name"] for x in ollama.list()["models"]])
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
            st.session_state["role_filter"] = role_filter
            st.session_state["temperature_filter"] = temperature_filter
            st.rerun()



class Oraculo:
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
        #if tools != []:
        #    model = initialize_agent(
        #        tools = load_tools(tools, allow_dangerous_tools = True), 
        #        llm = conversation,  # Pass the LLM chain with the prompt
        #        agent = "zero-shot-react-description",
        #        handle_parsing_errors = True
        #    )
        #else:
        #    model = conversation
        return conversation


class InformationRetrieval:
    def __init__(self):
        self.history = StreamlitChatMessageHistory(key = "chat_history")
        self.memory = ConversationBufferMemory(
            #memory_key = "chat_history", 
            return_messages = True,
            chat_memory = self.history)
    def load_model(self, tool_names, models_filter, temperature_filter):
        llm = OllamaLLM(
            model = models_filter,
            temperature = temperature_filter
        )
        if tool_names != []:
            tools = load_tools(
                tool_names = tool_names,
                llm = llm,
                allow_dangerous_tools = True
            )
            return initialize_agent(
                tools = tools,
                llm = llm,
                agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose = True,
                handle_parsing_errors = True
            )
        else:
            st.info("Choose at least one search engine tool.")
    

class SoftwareDevelopment:
    def __init__(self):
        self.history = StreamlitChatMessageHistory(key = "chat_messages")
        self.memory = ConversationBufferMemory(
            #memory_key = "chat_history", 
            return_messages = True,
            chat_memory = self.history)
    def load_model(self, models_filter, temperature_filter):
        llm = OllamaLLM(
            model = models_filter,
            temperature = temperature_filter)
        conversation = ConversationChain(
            llm = llm,
            #prompt = self.prompt,
            verbose = True,
            memory = self.memory
        )
        return initialize_agent(
            llm = conversation,
            tools = [
                PythonREPLTool(),
                ShellTool()],
            verbose = True,
            agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
    
class DataScience:
    def __init__(self, framework):
        self.history = StreamlitChatMessageHistory(key = "chat_messages")
        self.memory = ConversationBufferMemory(
            #memory_key = "chat_history", 
            return_messages = True,
            chat_memory = self.history)
        self.framework = framework
    def load_model(self, dataframe, models_filter, temperature_filter):
        llm = OllamaLLM(
            model = models_filter,
            temperature = temperature_filter)
        if self.framework == "LangChain":
            PROMPT = (
                "If you do not know the answer, say you don't know.\n"
                "Think step by step.\n"
                "\n"
                "Below is the query.\n"
                "Query: {query}\n"
            )
            prompt = PromptTemplate(
                template = PROMPT, 
                input_variables = ["query"])
            return create_pandas_dataframe_agent(
                llm,
                dataframe,
                verbose = True,
                allow_dangerous_code = True
            )
        elif self.framework == "PandasAI":
            return SmartDataframe(
                dataframe,
                config = {"llm": llm}
            )
    

class PlanAndSolve:
    def __init__(self):
        self.history = StreamlitChatMessageHistory(key = "chat_messages")
        self.memory = ConversationBufferMemory(
            #memory_key = "chat_history", 
            return_messages = True,
            chat_memory = self.history)
    def load_model(self, tool_names, models_filter, temperature_filter):
        llm = OllamaLLM(
            model = models_filter,
            temperature = temperature_filter)
        tools = load_tools(
            tool_names = tool_names,
            llm = llm,
            allow_dangerous_tools = True
        )
        planner = load_chat_planner(llm)
        executor = load_agent_executor(
            llm,
            tools,
            verbose = True
        )
        return PlanAndExecute(
            planner = planner,
            executor = executor,
            verbose = True
        )


class PDFAssistant:
    def __init__(self, llm, model_name, uploaded_file):
        self.llm = llm
        self.uploaded_file = uploaded_file
        self.embeddings = OllamaEmbeddings(model = model_name)
    def pdf_read(self, pdf_doc):
        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
            file.write(pdf_doc.getvalue())
            file_name = pdf_doc.name
        loader = UnstructuredFileLoader(temp_file, strategy = "fast")
        data = loader.load()
        text = ""
        for document in data:
            text += document.page_content
        os.remove("temp.pdf")
        return text
    def get_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000, chunk_overlap = 0)
        chunks = text_splitter.split_text(text)
        return chunks
    def vector_store(self, text_chunks):
        vector_store = Chroma.from_texts(
            text_chunks, embedding = self.embeddings)
        vector_store.save_local("chroma_db")
    def load_model(self, tools):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
"""You are a helpful assistant. Answer the question as detailed 
as possible from the provided context, 
make sure to provide all the details, 
if the answer is not in provided context just say, 
"answer is not available in the context", don't provide the wrong answer""",
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(
            self.llm, 
            [tools], 
            prompt)
        if tools is not None:
            agent_executor = AgentExecutor(
                agent = agent, 
                tools = [tools], 
                verbose = True)
        else:
            agent_executor = AgentExecutor(
                agent = agent, 
                verbose = True)
        return agent_executor
    def process_user_input(self, user_question):
        new_db = Chroma.load_local(
            "chroma_db", 
            self.embeddings,
            allow_dangerous_deserialization = True)
        retriever = new_db.as_retriever()
        retrieval_chain = create_retriever_tool(
            retriever,
            "pdf_extractor",
            "This tool is to give answer to queries from the pdf")
        return retrieval_chain

