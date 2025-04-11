from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY") # Retrieves API key from .env
os.environ["OPENAI_API_KEY"] = api_key # Set API key in the current environment

class RAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None

    def load_text_and_embed(self, raw_text: str):
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # Split texst into 500 tokens
        docs = splitter.create_documents([raw_text])
        
        # 2. Create FAISS (vector database) index
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def answer_question(self, query: str) -> str:
        if not self.vectorstore:
            return "No documents loaded."

        # 3. RAG with RetrievalQA
        retriever = self.vectorstore.as_retriever() # Converts vs. into retriever object (similarity search)
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4"), retriever=retriever) # Pipeline to 1. embed user q ; 2. feed q and retriever into GPT ; 3. Return a response
        return qa.run(query) # Run the pipeline on the user's query
