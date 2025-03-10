![COELHO GenAI](assets/coelho_genai_logo.png)
# COELHO GenAI
Demonstration: [COELHO GenAI Presentation](./COELHOGenAI.pdf)

COELHO GenAI is a platform that connects the user to open source Large Language Models like Llama (Meta), Gemma 2 (Google), Phi 3.5, Qwen 2.5, DeepSeek R1, OpenAI models (4o, 4o mini, o1, o3 etc.) and others, allowing users to have their own Language Model interface using the following tools:


**1) Assistant:** Assistant is a simple chatbot that can answer the questions you have in order to solve problems, have new thoughts and ideas, build new ideas and so on.  

**2) Information Retrieval:** Information Retrieval connects user local LLM to online tools, such as DuckDuckGo, Wikipedia, PubMed etc.  

**3) Data Science:** Data Science tool allows user to use autonomous AI agents to explore and to make data analysis and data science over data the user supplies to LLM.  

**4) Document Assistant:** Document Assistant is a powerful tool that gives you the power to analyze documents using Docling and lots of LangChain document loaders, like Wikipedia and dozens of other useful and famous services.  

**5) Plan & Solve:** Plan & Solve tool user AI agents to transform the user request into a detailed strategy planner to solve problems given by the user.


---

## Details about the project  

COELHO GenAI allows you to use 5 different APIs services to function with AI Agents, which 4 of them you need to get an API key in order to use the project:  
- [Groq](https://console.groq.com)
- [Ollama](https://ollama.com/download)
- [SambaNova](https://cloud.sambanova.ai/)
- [ScaleWay](https://account.scaleway.com/)
- [OpenAI](https://platform.openai.com/)

About Ollama, you can install LLM local models through [Ollama Models](https://ollama.com/library) or [HuggingFace Models](https://huggingface.co/models).

---

## How to install this project

1) Clone this repository:  
> git clone https://github.com/rafaelcoelho1409/COELHOGenAI  
2) Enter this repository folder:  
> cd COELHOGenAI  
3) Install UV for Python libraries management - [UV install](https://docs.astral.sh/uv/getting-started/installation/)  
4) Install Qdrant on Docker to enable vector database and RAG on platform to use Document Assistant - [Qdrant](https://qdrant.tech/documentation/quickstart/)  
> docker pull qdrant/qdrant  
> docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant  
5) Set a virtual environment and install requeriments  
> uv venv  
> source .venv/bin/activate  
> uv pip install -r requirements.txt
6) Finally, run COELHO GenAI on Streamlit:  
> streamlit run app.py