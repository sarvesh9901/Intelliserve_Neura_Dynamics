# IntelliAssist AI Chatbot

## 📌 About the Project
IntelliAssist AI Chatbot is an intelligent conversational assistant built using **LangChain**, **LangGraph**, and **Google Generative AI**.  
It can:
- Fetch real-time weather information 🌤
- Retrieve domain-specific answers from a **Pinecone vector database**
- Maintain conversation context for more natural interactions
- Run as a **Streamlit** web application for an interactive UI
- Be tested and evaluated using **LangSmith** and **Pytest**

Evaluation screenshots are available in the **Evaluation results** folder.  
Unit test execution results are stored in **test_result.md**.

---

## 🚀 Instructions to Run the Project

1️⃣ **Install `uv` (a fast Python package manager) and create virtual environment**  
```bash
pip install uv
uv venv


## 2️⃣ **Activate the virtual environment

**Windows (PowerShell)**  
```bash
.venv\Scripts\Activate

## 4️⃣ Install All Dependencies
```bash
uv add -r requirements.txt

## 5️⃣ Run the Streamlit App
```bash
streamlit run app.py


## 📂 Important Notes

- All API keys are already configured in the **.env** file → **Use them as they are**.
- **Evaluation screenshots** are available in the **Evaluation results** folder.
- **Unit test results** are saved in **test_result.md** for reference.

## 🛠️ Tech Stack

- **LangChain** – for building conversational logic and tool integration  
- **LangGraph** – for defining stateful conversational flows  
- **Google Generative AI** – for generating intelligent responses  
- **Pinecone** – for vector database retrieval  
- **Streamlit** – for web-based chat interface  
- **Pytest** – for unit testing  
- **LangSmith** – for evaluation and debugging  
