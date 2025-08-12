import os
import requests
from langchain.chains import ConversationalRetrievalChain
#from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# Function to create embeddings of PDF document using Hugging Face
def initiate_embedding_process(
    pdf_path=r"data\fact-vs-dimension-table-star-vs-snowflake-schema-data-import.pdf",
    chunk_size=500,
    chunk_overlap=50
):
    print("Starting the embedding process...")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    data = loader.load_and_split()
    print(f"Loaded {len(data)} documents.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = loader.load_and_split(text_splitter=splitter)
    print(f"Split into {len(chunks)} chunks.")

    # Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"  # You can change model here
    )

    # Store in Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=pinecone_index_name
    )
    print("Embedding process completed successfully.")

def get_conversation_chain(vector_search):
    # You can still use Gemini or GPT here, I kept GPT-4o from your code
    
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature= 0)
    memory = ConversationBufferWindowMemory(
        k=3, memory_key="chat_history", return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_search.as_retriever(),
        memory=memory
    )


# Function to find the category and subcategory from keywords
def find_match(question):
    query = f"""
    You are a knowledgeable assistant. Use the provided context from the database to answer the question accurately.
    Question: {question}
    - If the answer is not present in the database/context, respond exactly with: "I don't know."
    - Do not make up answers or add extra details outside the provided context.
    """

    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX')

    # Use the same Hugging Face model as you did in embedding upload
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"  # 768-dim to match Pinecone index
    )

    # Load existing Pinecone index
    vector_search = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    # Create conversation chain
    conversation_chain = get_conversation_chain(vector_search)

    # Run the query
    response = conversation_chain({"question": query})

    return response.get('answer', "")




if __name__ == "__main__":
    #initiate_embedding_process()
    #print(find_match("what is fact table ?"))
    pass
