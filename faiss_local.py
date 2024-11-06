
from langchain_community.embeddings import LlamaCppEmbeddings
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import re
import time
import os

db_faiss_path = "vectorstores/faiss"

class CustomLlamaCppEmbeddings(LlamaCppEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the Llama model.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        embeddings = [self.client.embed(text)[0] for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the Llama model.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        embedding = self.client.embed(text)[0]
        return list(map(float, embedding))


def text_clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\n+|\s+', ' ', text)
    text = re.sub(r' - ', '', text)
    return text

def clean_docs(docs: list[Document]) -> list[Document]:
    for doc in docs:
        doc.page_content = text_clean(doc.page_content)
    return docs

def load_pdf(file_path: str) -> list[str]:
    loader = PyPDFLoader(file_path)
    data = loader.load()
    cleaned_data = clean_docs(data)
    return cleaned_data

def save_upload_file(upload_file, destination: str):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as buffer:
        buffer.write(upload_file.file.read())

def process_documents(file_path: str) -> list[Document]:
    
    data_load = load_pdf(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = []
    for doc in data_load:
        chunks = text_splitter.split_text(doc.page_content)
        split_documents.extend([Document(page_content=chunk) for chunk in chunks])
    return split_documents

def create_faiss_db(documents: list[Document]):
    #embeddings = OllamaEmbeddings(model=model_name)
    embeddings =  CustomLlamaCppEmbeddings(model_path="./Llama-3.2-3B.Q3_K_S.gguf")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(db_faiss_path)
    print("FAISS database created")

if __name__ == "__main__":
    file_path = "./data/10 pages.pdf"
    documents = process_documents(file_path)
    db_start_time = time.time()  # Record start time
    print("Creating FAISS database...")
    create_faiss_db(documents)
    db_end_time = time.time()  # Record end time
    db_elapsed_time = db_end_time - db_start_time
    print(f"FAISS database created in {db_elapsed_time:.4f} seconds.")

