
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import LlamaCppEmbeddings
from typing import List
from langchain import hub
from langchain.chains import create_retrieval_chain


# AWS Lambda configuration
MODEL_NAME = "Llama-3.2-3B.Q3_K_S.gguf"
db_faiss_path = "vectorstores/faiss"
model_path = f"./model/{MODEL_NAME}"

# Lambda handler function
def handler(event):
    query = event['query']
    # Process the query
    response = chatbot(query)
    return {"response": response}
# Lambda handler function


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

# Chatbot
def chatbot(query:str)-> str:
    llm = LlamaCpp(model_path=model_path,temperature=0.75,max_tokens=500,top_p=1,verbose=True)
    embedder = CustomLlamaCppEmbeddings(model_path=model_path)
    db = FAISS.load_local(db_faiss_path,embeddings=embedder,allow_dangerous_deserialization=True,)
    # Create a retriever and document chain
    retriever = db.as_retriever()
    # Execute the chain with the retrieved documents
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    result = retrieval_chain.invoke({"input": query})
    return result

if __name__ == "__main__":
    query = "What is heat rash?"
    # Process the query
    response = chatbot(query)
    print(response)
    
