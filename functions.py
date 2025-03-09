import streamlit as st
import pandas as pd
import base64
import ollama
import os
import io
import json
import subprocess
from uuid import uuid4
#from pandasai import SmartDataframe
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.tools import ShellTool
from langchain_community.utilities import WikipediaAPIWrapper
#from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.tools import Tool
from langchain_core.documents import Document
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain.chains import LLMChain
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import (
    AgentType, 
    initialize_agent,
)
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.plan_and_execute import (
    load_chat_planner,
    load_agent_executor,
    PlanAndExecute
)
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langgraph.checkpoint.memory import MemorySaver



#>>>-------------------------------------------------<<<
#STREAMLIT
#>>>-------------------------------------------------<<<
@st.dialog("Settings", width = "large")
def settings():
    api_keys_dict = {
        "Groq": "GROQ_API_KEY",
        "SambaNova": "SAMBANOVA_API_KEY",
        "Scaleway": (
            "SCW_GENERATIVE_APIs_ENDPOINT",
            "SCW_ACCESS_KEY",
            "SCW_SECRET_KEY"
        ),
        "OpenAI": "OPENAI_API_KEY"
    }
    framework_option = st.selectbox(
        label = "Framework",
        options = [
            "Groq",
            #"Google Generative AI",
            "Ollama",
            "SambaNova",
            "Scaleway",
            "OpenAI"
        ]
    )
    st.session_state["framework"] = framework_option
    provider_model_dict = {
        "Groq": [
            "gemma2-9b-it",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-guard-3-8b",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "qwen-2.5-32b",
            "deepseek-r1-distill-qwen-32b",
            "deepseek-r1-distill-llama-70b-specdec",
            "deepseek-r1-distill-llama-70b",
            "llama-3.3-70b-specdec",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
                ], 
        "Google Generative AI": [
            "gemini-1.5-pro",
            #"gemini-2.0-flash"
        ],
        "SambaNova": [
            "DeepSeek-R1",
            "DeepSeek-R1-Distill-Llama-70B",
            "Llama-3.1-Tulu-3-405B",
            "Meta-Llama-3.1-405B-Instruct",
            "Meta-Llama-3.1-70B-Instruct",
            "Meta-Llama-3.1-8B-Instruct",
            "Meta-Llama-3.3-70B-Instruct",
            "Meta-Llama-Guard-3-8B",
            "Qwen2.5-72B-Instruct",
            "Qwen2.5-Coder-32B-Instruct",
            "QwQ-32B-Preview"
        ],
        "Scaleway": [
            "deepseek-r1",
            "deepseek-r1-distill-llama-70b",
            "llama-3.3-70b-instruct",
            "llama-3.1-70b-instruct",
            "llama-3.1-8b-instruct",
            "mistral-nemo-instruct-2407",
            "pixtral-12b-2409",
            "qwen2.5-coder-32b-instruct",
            "bge-multilingual-gemma2"
        ],
        "OpenAI": [
            "gpt-4o",
            "chatgpt-4o-latest",
            "gpt-4o-mini",
            "o1",
            "o1-mini",
            "o3-mini",
            "o1-preview"
        ]
    }
    if framework_option == "Ollama":
        with st.form("Settings Ollama"):
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
            toggle_filters = st.columns(3)
            try:
                memory_filter = toggle_filters[0].toggle(
                    label = "Memory",
                    value = st.session_state["memory_filter"]
                )
            except:
                memory_filter = toggle_filters[0].toggle(
                    label = "Memory",
                    value = True
                )
            try:
                vector_database_filter = toggle_filters[1].toggle(
                    label = "Vector database",
                    value = st.session_state["vector_database_filter"]
                )
            except:
                vector_database_filter = toggle_filters[1].toggle(
                    label = "Vector database",
                )
            try:
                rag_filter = toggle_filters[2].toggle(
                    label = "RAG",
                    value = st.session_state["rag_filter"]
                )
            except:
                rag_filter = toggle_filters[2].toggle(
                    label = "RAG",
                )
            submit_button = st.form_submit_button(
                    label = "Run model",
                    use_container_width = True
                )
            if submit_button:
                if "model_name" in st.session_state:
                    if st.session_state["model_name"] != models_filter:
                        subprocess.run([
                            "ollama",
                            "stop",
                            st.session_state["model_name"]
                        ],
                        )
                else:
                    subprocess.run([
                        "ollama",
                        "stop",
                        models_filter
                    ],
                    )
                st.session_state["model_name"] = models_filter
                st.session_state["temperature_filter"] = temperature_filter
                st.session_state["memory_filter"] = memory_filter
                st.session_state["vector_database_filter"] = vector_database_filter
                st.session_state["rag_filter"] = rag_filter
                st.rerun()
    elif framework_option in [
        "Groq",
        #"Google Generative AI",
        "SambaNova",
        "Scaleway",
        "OpenAI"
    ]:
        with st.form(f"Settings {framework_option}"):
            models_option = st.selectbox(
                label = f"{framework_option} Models", 
                options = provider_model_dict[framework_option])
            temperature_filter = st.slider(
                label = "Temperature",
                min_value = 0.00,
                max_value = 1.00,
                value = 0.00,
                step = 0.01
            )
            if st.session_state["framework"] in [
                "Groq",
                #"Google Generative AI",
                "SambaNova",
                "OpenAI"
            ]:
                #AUTOFILL API KEYS, IF EXISTS
                if api_keys_dict[st.session_state["framework"]] in os.environ:
                    globals()[api_keys_dict[st.session_state["framework"]]] = st.text_input(
                        label = api_keys_dict[st.session_state["framework"]],
                        value = os.getenv(api_keys_dict[st.session_state["framework"]]),
                        placeholder = "Provide the API key",
                        type = "password"
                    )
                    os.environ[api_keys_dict[st.session_state["framework"]]] = globals()[api_keys_dict[st.session_state["framework"]]]
                else:
                    globals()[api_keys_dict[st.session_state["framework"]]] = st.text_input(
                        label = api_keys_dict[st.session_state["framework"]],
                        #value = os.getenv(api_keys_dict[st.session_state["framework"]])
                        placeholder = "Provide the API key",
                        type = "password"
                    )
                    os.environ[api_keys_dict[st.session_state["framework"]]] = globals()[api_keys_dict[st.session_state["framework"]]]
            elif st.session_state["framework"] == "Scaleway":
                if "SCW_GENERATIVE_APIs_ENDPOINT" in os.environ:
                    SCW_GENERATIVE_APIs_ENDPOINT = st.text_input(
                        label = "SCW_GENERATIVE_APIs_ENDPOINT",
                        value = os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
                        placeholder = "Provide the API endpoint",
                        type = "password"
                    )
                else:
                    SCW_GENERATIVE_APIs_ENDPOINT = st.text_input(
                        label = "SCW_GENERATIVE_APIs_ENDPOINT",
                        placeholder = "Provide the API endpoint",
                        type = "password"
                    )
                if "SCW_ACCESS_KEY" in os.environ:
                    SCW_ACCESS_KEY = st.text_input(
                        label = "SCW_ACCESS_KEY",
                        value = os.getenv("SCW_ACCESS_KEY"),
                        placeholder = "Provide the access key",
                        type = "password"
                    )
                else:
                    SCW_ACCESS_KEY = st.text_input(
                        label = "SCW_ACCESS_KEY",
                        placeholder = "Provide the access key",
                        type = "password"
                    )
                if "SCW_SECRET_KEY" in os.environ:
                    SCW_SECRET_KEY = st.text_input(
                        label = "SCW_SECRET_KEY",
                        value = os.getenv("SCW_SECRET_KEY"),
                        placeholder = "Provide the secret key",
                        type = "password"
                    )
                else:
                    SCW_SECRET_KEY = st.text_input(
                        label = "SCW_SECRET_KEY",
                        placeholder = "Provide the secret key",
                        type = "password"
                    )
                os.environ["SCW_GENERATIVE_APIs_ENDPOINT"] = SCW_GENERATIVE_APIs_ENDPOINT
                os.environ["SCW_ACCESS_KEY"] = SCW_ACCESS_KEY
                os.environ["SCW_SECRET_KEY"] = SCW_SECRET_KEY
            toggle_filters = st.columns(3)
            try:
                memory_filter = toggle_filters[0].toggle(
                    label = "Memory",
                    value = st.session_state["memory_filter"]
                )
            except:
                memory_filter = toggle_filters[0].toggle(
                    label = "Memory",
                    value = True
                )
            try:
                vector_database_filter = toggle_filters[1].toggle(
                    label = "Vector database",
                    value = st.session_state["vector_database_filter"]
                )
            except:
                vector_database_filter = toggle_filters[1].toggle(
                    label = "Vector database",
                )
            try:
                rag_filter = toggle_filters[2].toggle(
                    label = "RAG",
                    value = st.session_state["rag_filter"]
                )
            except:
                rag_filter = toggle_filters[2].toggle(
                    label = "RAG",
                )
            submit_button = st.form_submit_button(
                    label = "Run model",
                    use_container_width = True
                )
            if submit_button:
                st.session_state["model_name"] = models_option
                st.session_state["temperature_filter"] = temperature_filter
                st.session_state["memory_filter"] = memory_filter
                st.session_state["vector_database_filter"] = vector_database_filter
                st.session_state["rag_filter"] = rag_filter
                st.rerun()

@st.dialog("Prompt settings", width = "large")
def prompt_settings():
    with st.form("LangChain Hub"):
        PROMPT_NAME = st.text_input(
            label = "Prompt name (LangChain Hub)"
        )
        st.caption("**Example: hardkothari/prompt-maker**")
        prompt_name_submit = st.form_submit_button(
            label = "Run prompt",
            use_container_width = True
        )
        st.divider()
        st.link_button(
            "LangChain Hub",
            "https://smith.langchain.com/hub",
            use_container_width = True
        )
        if prompt_name_submit:
            try:
                PROMPT = hub.pull(PROMPT_NAME)
                st.session_state["PROMPT_NAME"] = PROMPT_NAME
                st.session_state["PROMPT"] = PROMPT
            except:
                st.error("Invalid prompt name.")
                st.stop()
            st.rerun()

@st.dialog("Prompt informations", width = "large")
def prompt_informations(PROMPT_NAME, PROMPT):
    st.markdown(f"**Prompt name:** {PROMPT_NAME}")
    st.divider()
    st.markdown("**Prompt template:**")
    prompt_description = ""
    prompt_agent = ["SYSTEM", "HUMAN"]
    if type(PROMPT) in [ChatPromptTemplate, StructuredPrompt]:
        for i in range(len(PROMPT.messages)):
            prompt_description += f"({prompt_agent[i]})\n\n"
            try:
                prompt_description += PROMPT.messages[i].prompt.template
            except:
                prompt_description += PROMPT.messages[i].content
            prompt_description += "\n\n"
    elif type(PROMPT) == PromptTemplate:
        st.write(PROMPT.template)
    st.markdown(prompt_description)

@st.dialog("Retrieved documents", width = "large")
def retrieved_documents(processed_doc, loader_framework):
    st.markdown(f"**Retrieved documents**")
    st.divider()
    if loader_framework == "Docling":
        rag_result = processed_doc.document.export_to_markdown()
    elif loader_framework == "LangChain":
        rag_result = "\n\n".join(x.page_content for x in processed_doc)
    elif loader_framework == "Youtube":
        rag_result = processed_doc
    st.write(rag_result)


@st.dialog("Application graph", width = "large")
def view_application_graph(graph):
    st.image(graph.get_graph().draw_mermaid_png())



@st.cache_resource
def docling_process_document(
    document_type, 
    uploaded_file_content = None, 
    uploaded_file_name = None,
    url = None
    ):
    if document_type == "File":
        buffered = io.BytesIO(uploaded_file_content)
        content = DocumentStream(
            name = uploaded_file_name,
            stream = buffered
        )
        with st.spinner("Converting file"):
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            doc_converter = DocumentConverter(
                allowed_formats = [
                    InputFormat.PDF,
                    InputFormat.IMAGE,
                    InputFormat.DOCX,
                    InputFormat.HTML,
                    InputFormat.PPTX,
                    InputFormat.ASCIIDOC,
                    InputFormat.MD,
                ]
            )
            result = doc_converter.convert(
                content
            )
            return result
    elif document_type == "URL":
        doc_converter = DocumentConverter()
        result = doc_converter.convert(url)
        return result
    

@st.cache_resource
def docling_save_artifacts(_processed_doc):
    with st.spinner("Saving artifacts"):
        processed_doc_dict = _processed_doc.document.export_to_dict()
        processed_doc_md = _processed_doc.document.export_to_markdown()
        for x in [
            f"docling/documents/{processed_doc_dict['name']}",
            f"docling/documents/{processed_doc_dict['name']}/images",
            f"docling/documents/{processed_doc_dict['name']}/tables"
        ]:
            try:
                os.makedirs(x)
            except:
                pass
    with st.spinner("Saving document in markdown"):
        #save doc in markdown
        with open(f"docling/documents/{processed_doc_dict['name']}/{processed_doc_dict['name']}.md", "w") as outfile:
            outfile.write(processed_doc_md)
    with st.spinner("Saving document in JSON"):
        #save doc in JSON
        with open(f"docling/documents/{processed_doc_dict['name']}/{processed_doc_dict['name']}.json", "w") as outfile:
            json.dump(processed_doc_dict, outfile)
    with st.spinner("Saving images from document"):
        #save page images
        for page_no, page in _processed_doc.document.pages.items():
            page_no = page.page_no
            page_image_filename = f"docling/documents/{processed_doc_dict['name']}/images/{page_no}.png"
            try:
                with open(page_image_filename, "wb") as outfile:
                    page.image.pil_image.save(outfile, format = "PNG")
            except:
                pass
    with st.spinner("Saving images of figures and tables from document"):
        #save images of figures and tables
        table_counter = 0
        picture_counter = 0
        for element, _level in _processed_doc.document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                element_image_filename = f"docling/documents/{processed_doc_dict['name']}/images/table-{table_counter}.png"
                try:
                    with open(element_image_filename, "wb") as outfile:
                        element.get_image(_processed_doc.document).save(outfile, format = "PNG")
                except:
                    pass
            if isinstance(element, PictureItem):
                picture_counter += 1
                element_image_filename = f"docling/documents/{processed_doc_dict['name']}/images/picture-{picture_counter}.png"
                try:
                    with open(element_image_filename, "wb") as outfile:
                        element.get_image(_processed_doc.document).save(outfile, "PNG")
                except:
                    pass
    with st.spinner("Saving document in Markdown and HTML"):
        try:
            # Save markdown with embedded pictures
            md_filename = f"docling/documents/{processed_doc_dict['name']}-with-images.md"
            _processed_doc.document.save_as_markdown(md_filename, image_mode = ImageRefMode.EMBEDDED)
        except:
            pass
        try:
            # Save markdown with externally referenced pictures
            md_filename = f"docling/documents/{processed_doc_dict['name']}-with-image-refs.md"
            _processed_doc.document.save_as_markdown(md_filename, image_mode = ImageRefMode.REFERENCED)
        except:
            pass            
        try:
            # Save HTML with externally referenced pictures
            html_filename = f"docling/documents/{processed_doc_dict['name']}-with-image-refs.html"
            _processed_doc.document.save_as_html(html_filename, image_mode = ImageRefMode.REFERENCED)
        except:
            pass
    with st.spinner("Saving tables from document"):
        # Export tables
        for table_ix, table in enumerate(_processed_doc.document.tables):
            table_df: pd.DataFrame = table.export_to_dataframe()
            # Save the table as csv
            element_csv_filename = f"docling/documents/{processed_doc_dict['name']}/tables/table-{table_ix+1}.csv"
            table_df.to_csv(element_csv_filename)
            # Save the table as html
            element_html_filename = f"docling/documents/{processed_doc_dict['name']}/tables/table-{table_ix+1}.html"
            with open(element_html_filename, "w") as fp:
                fp.write(table.export_to_html())


@st.cache_resource
def store_on_qdrant(_client, _processed_doc, model_name, loader_framework, llm_framework):
    if llm_framework == "Ollama":
        embeddings = OllamaEmbeddings(model = model_name)
    elif llm_framework == "Google Generative AI":
        embeddings = GoogleGenerativeAIEmbeddings(model = model_name)
    elif llm_framework in ["Groq", "SambaNova"]:
        embeddings = HuggingFaceEmbeddings(model = "all-MiniLM-L6-v2")
    elif llm_framework in ["Scaleway", "OpenAI"]:
        embeddings = OpenAIEmbeddings(model = model_name)
    embedding_vector = embeddings.embed_query("This is a test query")
    if not _client.collection_exists("document_assistant"):
        _client.create_collection(
            collection_name = "document_assistant",
            vectors_config = VectorParams(
                size = len(embedding_vector), 
                distance = Distance.COSINE)
        )
    #self.qdrant_client.delete_collection("document_assistant")
    vector_store = QdrantVectorStore(
        client = _client,
        collection_name = "document_assistant",
        embedding = embeddings,
    )
    if loader_framework == "Docling":
        document = Document(
            page_content = _processed_doc.document.export_to_markdown())
    elif loader_framework == "LangChain":
        document = Document(
            page_content = "\n\n".join(x.page_content for x in _processed_doc))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
    )
    splits = text_splitter.split_documents([document])
    ids = [str(uuid4()) for _ in range(len(splits))]
    vector_store.add_documents(documents = splits, ids = ids)
    return vector_store

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
    active_models_container = st.sidebar.container()
    active_models_text = "**Active model:** "
    if st.session_state["framework"] == "Ollama":
        if ollama.ps()["models"] != []:
            for model_name in ollama.ps()["models"]:
                active_models_text += f"* {model_name['model']}\n"
        else:
            active_models_text += "No active models."
    elif st.session_state["framework"] == "Groq":
        active_models_text += st.session_state["model_name"]
    active_models_container.info(active_models_text)

def check_model_and_temperature():
    return all([x in st.session_state.keys() for x in ["model_name", "temperature_filter"]])

def initialize_shared_memory():
    # Initialize shared memory
    if "history" not in st.session_state:
        st.session_state["history"] = StreamlitChatMessageHistory(key = "chat_history")
    if "shared_memory" not in st.session_state:
        st.session_state["shared_memory"] = ConversationBufferMemory(
            memory_key = "chat_history", 
            input_key = "input",
            return_messages = True,
            chat_memory = st.session_state["history"]
        )
    if "langgraph_memory" not in st.session_state:
        st.session_state["langgraph_memory"] = MemorySaver()


#>>>-------------------------------------------------<<<
#CLASSES
#>>>-------------------------------------------------<<<
class Assistant:
    def __init__(self):
        self.prompt_template = """
            You are a nice chatbot having a conversation with a human.
    
            Chat history:
            {chat_history}
    
            Human: {input}
            """
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)
    def load_model(self, framework, temperature_filter, model_name, memory):
        self.llm_framework = {
            "Groq": ChatGroq,
            "Ollama": ChatOllama,
            "Google Generative AI": ChatGoogleGenerativeAI,
            "SambaNova": ChatSambaNovaCloud,
            "Scaleway": ChatOpenAI,
            "OpenAI": ChatOpenAI,
        }
        self.llm_model = self.llm_framework[framework]
        if framework == "Scaleway":
            self.llm = ChatOpenAI(
                base_url = os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
                api_key = os.getenv("SCW_SECRET_KEY"),
                model = model_name,
                temperature =  temperature_filter
            )
        else:
            try:
                self.llm = self.llm_model(
                    model = model_name,
                    temperature = temperature_filter,
                )
            except:
                self.llm = self.llm_model(
                    model = model_name,
                    #temperature = temperature_filter,
                )
        conversation = ConversationChain(
            llm = self.llm,
            prompt = self.prompt,
            verbose = True,
            memory = memory
        )
        return conversation
    

class InformationRetrieval:
    def __init__(self):
        pass
    def load_model(self, framework, tools, model_name, temperature_filter, memory):
        self.llm_framework = {
            "Groq": ChatGroq,
            "Ollama": ChatOllama,
            "Google Generative AI": ChatGoogleGenerativeAI,
            "SambaNova": ChatSambaNovaCloud,
            "Scaleway": ChatOpenAI,
            "OpenAI": ChatOpenAI,
        }
        self.llm_model = self.llm_framework[framework]
        if framework == "Scaleway":
            self.llm = ChatOpenAI(
                base_url = os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
                api_key = os.getenv("SCW_SECRET_KEY"),
                model = model_name,
                temperature =  temperature_filter
            )
        else:
            try:
                self.llm = self.llm_model(
                    model = model_name,
                    temperature = temperature_filter,
                )
            except:
                self.llm = self.llm_model(
                    model = model_name,
                    #temperature = temperature_filter,
                )
        if tools != []:
            return initialize_agent(
                tools = tools,
                llm = self.llm,
                memory = memory,
                agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose = True,
                handle_parsing_errors = True,
                #max_iterations = 5
            )
        else:
            st.info("Choose at least one search engine tool.")

class DataScience:
    def __init__(self, framework):
        self.framework = framework
    def load_model(self, framework, dataframe, model_name, temperature_filter, memory):
        self.llm_framework = {
            "Groq": ChatGroq,
            "Ollama": ChatOllama,
            "Google Generative AI": ChatGoogleGenerativeAI,
            "SambaNova": ChatSambaNovaCloud,
            "Scaleway": ChatOpenAI,
            "OpenAI": ChatOpenAI,
        }
        self.llm_model = self.llm_framework[framework]
        if framework == "Scaleway":
            self.llm = ChatOpenAI(
                base_url = os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
                api_key = os.getenv("SCW_SECRET_KEY"),
                model = model_name,
                temperature =  temperature_filter
            )
        else:
            try:
                self.llm = self.llm_model(
                    model = model_name,
                    temperature = temperature_filter,
                )
            except:
                self.llm = self.llm_model(
                    model = model_name,
                    #temperature = temperature_filter,
                )
        #self.smartdataframe = SmartDataframe(
        #        dataframe,
        #        config = {"llm": self.llm}
        #    )
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
                self.llm,
                dataframe,
                memory = memory,
                verbose = True,
                allow_dangerous_code = True,
                agent_executor_kwargs = {
                    "handle_parsing_errors": True,
                    },
                #max_iterations = 5
            )
        #elif self.framework == "PandasAI":
        #    return self.smartdataframe
        
class PromptEngineering:
    def __init__(self, PROMPT):
        self.prompt = PROMPT
        if type(PROMPT) in [ChatPromptTemplate, StructuredPrompt]:
            self.human_input_variables = [
                x.input_variables for x in PROMPT.messages 
                if type(x) == HumanMessagePromptTemplate]
            self.system_input_variables = [
                x.input_variables for x in PROMPT.messages 
                if type(x) == SystemMessagePromptTemplate]
            if self.system_input_variables == []:
                self.input_variables = self.human_input_variables[0]
            elif self.human_input_variables == []:
                self.input_variables = self.system_input_variables[0]
            else:
                self.input_variables = self.system_input_variables[0] + self.human_input_variables[0]
        elif type(PROMPT) == PromptTemplate:
            self.input_variables = PROMPT.input_variables
    def load_model(self, models_filter, temperature_filter, memory):
        self.llm = ChatOllama(
            model = models_filter,
            temperature = temperature_filter
        )
        conversation = LLMChain(
            llm = self.llm,
            prompt = self.prompt,
            verbose = True,
            memory = memory
        )
        return conversation
    

class DocumentAssistant:
    def __init__(self, model_name, vector_database_filter):
        self.embeddings_dict = {
            "Groq": HuggingFaceEmbeddings,
            "Ollama": OllamaEmbeddings,
            "Google Generative AI": ChatGoogleGenerativeAI,
            "SambaNova": HuggingFaceEmbeddings,
            "Scaleway": ChatOpenAI,
            "OpenAI": ChatOpenAI,
        }
        #remove .lock file
        self.vector_database_path = {
            True: os.path.join(
                "databases",
                model_name),
            False: ":memory:"
        }
        #preventing qdrant error
        lock_file = os.path.join(model_name, ".lock")
        if os.path.exists(lock_file):
            os.remove(lock_file)
        #self.qdrant_client = QdrantClient(url = "http://localhost:6333") #>>running qdrant on docker
        self.qdrant_client = QdrantClient(path = self.vector_database_path[vector_database_filter])
        self.embeddings = OllamaEmbeddings(model = model_name)
        self.template = """
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Keep the answer concise.

            Context: {context}
            
            User question: {input}
            
            Previous conversation: {chat_history}
            """
        self.prompt = PromptTemplate(
            input_variables = [
                "context", 
                "chat_history", 
                "input"
                ],
            template = self.template
        )

    def load_model(self, framework, temperature_filter, model_name, memory, loader_framework):
        self.llm_framework = {
            "Groq": ChatGroq,
            "Ollama": ChatOllama,
            "Google Generative AI": ChatGoogleGenerativeAI,
            "SambaNova": ChatSambaNovaCloud,
            "Scaleway": ChatOpenAI,
            "OpenAI": ChatOpenAI,
        }
        self.llm_model = self.llm_framework[framework]
        if framework == "Scaleway":
            self.llm = ChatOpenAI(
                base_url = os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
                api_key = os.getenv("SCW_SECRET_KEY"),
                model = model_name,
                temperature =  temperature_filter
            )
        else:
            try:
                self.llm = self.llm_model(
                    model = model_name,
                    temperature = temperature_filter,
                )
            except:
                self.llm = self.llm_model(
                    model = model_name,
                    #temperature = temperature_filter,
                )
        conversation = LLMChain(
            llm = self.llm,
            prompt = self.prompt,
            verbose = True,
            memory = memory,
        )
        return conversation
        
        
    
class SoftwareDevelopment:
    def __init__(self):
        pass
    def load_model(self, framework, model_name, temperature_filter, memory):
        self.llm_framework = {
            "Groq": ChatGroq,
            "Ollama": ChatOllama,
            "Google Generative AI": ChatGoogleGenerativeAI,
            "SambaNova": ChatSambaNovaCloud,
            "Scaleway": ChatOpenAI,
            "OpenAI": ChatOpenAI,
        }
        self.llm_model = self.llm_framework[framework]
        if framework == "Scaleway":
            self.llm = ChatOpenAI(
                base_url = os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
                api_key = os.getenv("SCW_SECRET_KEY"),
                model = model_name,
                temperature =  temperature_filter
            )
        else:
            try:
                self.llm = self.llm_model(
                    model = model_name,
                    temperature = temperature_filter,
                )
            except:
                self.llm = self.llm_model(
                    model = model_name,
                    #temperature = temperature_filter,
                )
        return initialize_agent(
            llm = self.llm,
            memory = memory,
            tools = [
                ShellTool(),
                PythonREPLTool(),
                ],
            verbose = True,
            agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            #max_iterations = 5
        )
    
class PlanAndSolve:
    def __init__(self):
        pass
    def load_model(self, framework, model_name, temperature_filter, memory):
        self.llm_framework = {
            "Groq": ChatGroq,
            "Ollama": ChatOllama,
            "Google Generative AI": ChatGoogleGenerativeAI,
            "SambaNova": ChatSambaNovaCloud,
            "Scaleway": ChatOpenAI,
            "OpenAI": ChatOpenAI,
        }
        self.llm_model = self.llm_framework[framework]
        if framework == "Scaleway":
            self.llm = ChatOpenAI(
                base_url = os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
                api_key = os.getenv("SCW_SECRET_KEY"),
                model = model_name,
                temperature =  temperature_filter
            )
        else:
            try:
                self.llm = self.llm_model(
                    model = model_name,
                    temperature = temperature_filter,
                )
            except:
                self.llm = self.llm_model(
                    model = model_name,
                    #temperature = temperature_filter,
                )
        planner = load_chat_planner(self.llm)
        search = WikipediaAPIWrapper()
        tools = [
            Tool(
                name = "Search",
                func = search.run,
                description = "useful for when you need to answer questions about current events"
            ),
        ]
        executor = load_agent_executor(
            self.llm,
            tools,
            verbose = True
        )
        return PlanAndExecute(
            planner = planner,
            executor = executor,
            memory = memory,
            verbose = True
        )
    

